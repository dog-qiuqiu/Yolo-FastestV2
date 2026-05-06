"""
摄像头实时推理 Demo (NCNN backend, 不依赖 PyTorch)

特性:
- 仅依赖 numpy / opencv / ncnn / PIL, 体积极小, 易于迁移到嵌入式 / 树莓派
- 自动查找 weights/ 下的 .ncnn.param + 同名 .ncnn.bin
- stretch (默认) 或 letterbox 预处理
- 中文标签 + 中文 HUD (PIL 渲染微软雅黑/黑体)
- 解码 + per-class NMS 用纯 numpy 实现 (复用 infer_onnx.decode/nms)

用法:
    python demo_camera_ncnn.py
    python demo_camera_ncnn.py --param weights/dms_kaggle_sim.ncnn.param \\
                               --bin   weights/dms_kaggle_sim.ncnn.bin \\
                               --cam 0 --conf 0.5
    python demo_camera_ncnn.py --cam path/to/video.mp4

按键:
    Q / ESC : 退出
    S       : 保存当前帧到 snapshots/
    SPACE   : 暂停 / 继续
"""
import os
import glob
import time
import argparse

import cv2
import numpy as np
import ncnn
from PIL import Image, ImageDraw, ImageFont

import utils.utils
from infer_onnx import decode, nms, preprocess  # 这两个函数纯 numpy 实现, 可直接复用


# 类别固定配色 (BGR), 与 dms-kaggle.names 顺序对应
CLASS_COLORS = [
    (0, 255, 0),     # Open Eye    - 绿
    (0, 0, 255),     # Closed Eye  - 红
    (0, 165, 255),   # Cigarette   - 橙
    (255, 0, 255),   # Phone       - 品红
    (255, 200, 0),   # Seatbelt    - 青蓝
]

EN_TO_ZH = {
    "Open Eye":   "睁眼",
    "Closed Eye": "闭眼",
    "Cigarette":  "香烟",
    "Phone":      "手机",
    "Seatbelt":   "安全带",
}

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",   # Linux 备选
    "/System/Library/Fonts/PingFang.ttc",                # macOS 备选
]


# -------------------- 工具函数 --------------------

def find_ncnn_model(weights_dir: str):
    """在 weights_dir 下查找 *.ncnn.param + 同名 .ncnn.bin, 返回 (param_path, bin_path)."""
    params = glob.glob(os.path.join(weights_dir, "*.ncnn.param"))
    if not params:
        raise FileNotFoundError(
            f"在 {weights_dir} 下未找到 *.ncnn.param. 请先用 pnnx 转换:\n"
            f"  pnnx weights/<your>.onnx inputshape=[1,3,352,352]"
        )
    # 选最近修改的
    param_path = max(params, key=os.path.getmtime)
    bin_path = param_path.replace('.param', '.bin')
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"找到 {param_path} 但缺少对应的 {bin_path}")
    return param_path, bin_path


def get_chinese_font(size: int, font_path: str = None):
    candidates = [font_path] if font_path else []
    candidates += FONT_CANDIDATES
    for p in candidates:
        if p and os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    print("[demo_ncnn] 警告: 未找到中文字体, 中文可能无法显示")
    return ImageFont.load_default()


def load_label_names(names_path: str):
    with open(names_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# -------------------- NCNN 推理封装 --------------------

class NCNNDetector:
    def __init__(self, param_path, bin_path, cfg, num_threads=4):
        self.cfg = cfg
        self.W = cfg['width']
        self.H = cfg['height']
        self.NC = cfg['classes']
        self.NA = cfg['anchor_num']
        self.anchors = np.array(cfg['anchors']).reshape(2, self.NA, 2)

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        self.num_threads = num_threads  # 仅作记录, ncnn-python 用全局默认

    def __call__(self, frame_bgr, conf_thres=0.5, iou_thres=0.4):
        """输入 BGR 帧, 返回原图坐标系的 dets [(x1,y1,x2,y2,score,cls_id), ...]."""
        ori_h, ori_w = frame_bgr.shape[:2]

        # preprocess (返回 (1,3,H,W)) -> ncnn 需要 (3,H,W) 且必须连续
        x = preprocess(frame_bgr, self.W, self.H)
        x_np = np.ascontiguousarray(x[0])

        x_mat = ncnn.Mat(x_np).clone()
        with self.net.create_extractor() as ex:
            ex.input('in0', x_mat)
            _, out0 = ex.extract('out0')
            _, out1 = ex.extract('out1')
            a0 = np.array(out0)
            a1 = np.array(out1)

        # decode 期望 (1, gh, gw, 20)
        outs = [a0[None], a1[None]]
        dets_all = []
        for i, o in enumerate(outs):
            gh = o.shape[1]
            stride = self.H / gh
            d = decode(o, self.anchors[i], stride, self.NC, self.NA, conf_thres=conf_thres)
            dets_all.append(d)
        dets = np.concatenate(dets_all, axis=0) if dets_all else np.zeros((0, 6))
        dets = nms(dets, iou_thres)

        # 眼睛类 (cls 0/1) 最多保留 2 个框, 其余类最多 1 个框
        EYE_CLS = {0, 1}
        MAX_PER_CLS = 2  # 眼睛上限
        if len(dets) > 0:
            keep = {}
            for d in sorted(dets, key=lambda x: -x[4]):  # score 从高到低
                cls_id = int(d[5])
                limit = MAX_PER_CLS if cls_id in EYE_CLS else 1
                bucket = keep.setdefault(cls_id, [])
                if len(bucket) < limit:
                    bucket.append(d)
            dets = np.array([d for bucket in keep.values() for d in bucket], dtype=np.float32)

        # 反映射到原图 (stretch 反变换)
        sx = ori_w / self.W
        sy = ori_h / self.H
        results = []
        for d in dets:
            x1, y1, x2, y2, sc, c = d
            x1 = int(max(0, min(ori_w - 1, x1 * sx)))
            y1 = int(max(0, min(ori_h - 1, y1 * sy)))
            x2 = int(max(0, min(ori_w - 1, x2 * sx)))
            y2 = int(max(0, min(ori_h - 1, y2 * sy)))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append((x1, y1, x2, y2, float(sc), int(c)))
        return results

    def close(self):
        # 显式释放 (Python GC 也会处理, 但提前释放更稳)
        del self.net


# -------------------- 绘制 --------------------

def draw_boxes_cv(frame, detections):
    for x1, y1, x2, y2, _, c in detections:
        color = CLASS_COLORS[c % len(CLASS_COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def draw_texts_pil(frame_bgr, detections, hud_lines, en_names, zh_names,
                   font, font_small):
    """PIL 一次性渲染所有中文 (检测框标签 + 左上 HUD + 底部提示)."""
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 检测框标签
    for x1, y1, x2, y2, sc, c in detections:
        color_bgr = CLASS_COLORS[c % len(CLASS_COLORS)]
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        name = zh_names[c] if c < len(zh_names) else f"cls{c}"
        text = f"{name} {sc:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = x1, y1 - th - 6
        if y < 0:
            y = y1 + 2
        draw.rectangle([x, y, x + tw + 6, y + th + 6], fill=color_rgb)
        draw.text((x + 3, y + 2), text, font=font, fill=(0, 0, 0))

    # 左上 HUD
    y = 8
    for line in hud_lines:
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((10 + dx, y + dy), line, font=font, fill=(0, 0, 0))
        draw.text((10, y), line, font=font, fill=(255, 255, 255))
        bbox = draw.textbbox((0, 0), line, font=font)
        y += (bbox[3] - bbox[1]) + 6

    # 底部提示
    hint = "Q/ESC 退出   S 截图   SPACE 暂停/继续"
    bbox = draw.textbbox((0, 0), hint, font=font_small)
    th = bbox[3] - bbox[1]
    yy = img_pil.size[1] - th - 10
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((10 + dx, yy + dy), hint, font=font_small, fill=(0, 0, 0))
    draw.text((10, yy), hint, font=font_small, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# -------------------- 主流程 --------------------

def parse_args():
    p = argparse.ArgumentParser(description="Yolo-FastestV2 摄像头实时推理 (NCNN 后端, 中文显示)")
    p.add_argument("--data", type=str, default="data/dms-kaggle.data",
                   help="训练配置 *.data 路径")
    p.add_argument("--param", type=str, default="",
                   help=".ncnn.param 路径; 留空则自动查找 weights/*.ncnn.param")
    p.add_argument("--bin", type=str, default="",
                   help=".ncnn.bin 路径; 留空则与 --param 同名替换扩展名")
    p.add_argument("--weights-dir", type=str, default="weights",
                   help="自动查找 ncnn 模型时使用的目录")
    p.add_argument("--cam", default="0", help="摄像头索引或视频文件路径")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.4, help="NMS IoU 阈值")
    p.add_argument("--width", type=int, default=0, help="摄像头采集宽度 (0=默认)")
    p.add_argument("--height", type=int, default=0, help="摄像头采集高度 (0=默认)")
    p.add_argument("--threads", type=int, default=4, help="ncnn 线程数 (仅记录用)")
    p.add_argument("--font", type=str, default="", help="中文字体路径")
    p.add_argument("--font-size", type=int, default=22, help="文字字号")
    p.add_argument("--snapshot-dir", type=str, default="snapshots",
                   help="按 S 截图的保存目录")
    return p.parse_args()


def open_capture(cam_arg, want_w, want_h):
    if cam_arg.isdigit():
        cap = cv2.VideoCapture(int(cam_arg), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(cam_arg))
    else:
        if not os.path.exists(cam_arg):
            raise FileNotFoundError(f"视频文件不存在: {cam_arg}")
        cap = cv2.VideoCapture(cam_arg)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开输入源: {cam_arg}")
    if want_w > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
    if want_h > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
    return cap


def main():
    opt = parse_args()
    assert os.path.exists(opt.data), f"找不到配置文件: {opt.data}"
    cfg = utils.utils.load_datafile(opt.data)

    # 解析 ncnn 模型路径
    if opt.param:
        param_path = opt.param
        bin_path = opt.bin or param_path.replace('.param', '.bin')
        assert os.path.exists(param_path), f"找不到: {param_path}"
        assert os.path.exists(bin_path), f"找不到: {bin_path}"
    else:
        param_path, bin_path = find_ncnn_model(opt.weights_dir)

    print(f"[demo_ncnn] backend     = NCNN")
    print(f"[demo_ncnn] data config = {opt.data}")
    print(f"[demo_ncnn] param       = {param_path}")
    print(f"[demo_ncnn] bin         = {bin_path}")
    print(f"[demo_ncnn] input size  = {cfg['width']} x {cfg['height']}")
    print(f"[demo_ncnn] conf / iou  = {opt.conf} / {opt.iou}")

    # 加载模型
    detector = NCNNDetector(param_path, bin_path, cfg, num_threads=opt.threads)

    # 标签
    en_names = load_label_names(cfg["names"])
    zh_names = [EN_TO_ZH.get(n, n) for n in en_names]
    print(f"[demo_ncnn] classes     = {list(zip(en_names, zh_names))}")

    # 字体
    font = get_chinese_font(opt.font_size, opt.font or None)
    font_small = get_chinese_font(max(14, opt.font_size - 6), opt.font or None)

    # 摄像头
    cap = open_capture(opt.cam, opt.width, opt.height)
    os.makedirs(opt.snapshot_dir, exist_ok=True)

    # 预热: 跑几次空 frame 抹平首帧延迟
    print("[demo_ncnn] 预热中...")
    dummy = np.zeros((cfg['height'], cfg['width'], 3), dtype=np.uint8)
    for _ in range(3):
        detector(dummy, conf_thres=opt.conf, iou_thres=opt.iou)

    win_name = "Yolo-FastestV2 NCNN Camera Demo (CN)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    paused = False
    last_t = time.perf_counter()
    fps_smooth = 0.0
    last_render = None
    last_dets = []
    last_infer_ms = 0.0
    fail_count = 0

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok or frame is None:
                    fail_count += 1
                    if fail_count >= 3:
                        print("[demo_ncnn] 连续读帧失败 / 视频结束, 退出")
                        break
                    print(f"[demo_ncnn] 读取帧失败, 重试 ({fail_count}/3)...")
                    time.sleep(0.05)
                    continue
                fail_count = 0

                # 推理
                t0 = time.perf_counter()
                detections = detector(frame, conf_thres=opt.conf, iou_thres=opt.iou)
                t1 = time.perf_counter()
                last_infer_ms = (t1 - t0) * 1000.0
                last_dets = detections

                draw_boxes_cv(frame, detections)
                last_render = frame

            # FPS 平滑
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = inst_fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * inst_fps

            if last_render is None:
                continue

            hud_lines = [
                f"后端 BACKEND: NCNN",
                f"帧率 FPS: {fps_smooth:5.1f}",
                f"推理耗时: {last_infer_ms:5.1f} ms",
                f"检测数量: {len(last_dets)}",
            ]
            if paused:
                hud_lines.append("已暂停")

            display = draw_texts_pil(
                last_render.copy() if paused else last_render,
                last_dets, hud_lines, en_names, zh_names, font, font_small,
            )

            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(opt.snapshot_dir, f"ncnn_snap_{ts}.png")
                cv2.imwrite(save_path, display)
                print(f"[demo_ncnn] 已保存截图 -> {save_path}")
            elif key == ord(" "):
                paused = not paused
                print(f"[demo_ncnn] {'暂停' if paused else '继续'}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
