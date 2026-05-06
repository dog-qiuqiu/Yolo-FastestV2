"""
摄像头实时推理 Demo
默认加载 weights/ 目录下 mAP 最高的 dms-kaggle 权重，对本机摄像头画面进行实时检测。

特性:
- 等比缩放 (letterbox) 预处理，框坐标按反变换还原
- 默认置信度阈值 0.6
- 中文标签 + 中文 HUD (使用 PIL 渲染微软雅黑，避免 OpenCV 中文乱码)
- CUDA 预热、weights_only 安全加载、读帧失败重试、坐标越界裁剪、空检测兼容

用法:
    python demo_camera.py
    python demo_camera.py --cam 1 --conf 0.5
    python demo_camera.py --weights weights/dms-kaggle-150-epoch-0.699052ap-model.pth
    python demo_camera.py --cam path/to/video.mp4

按键:
    Q / ESC : 退出
    S       : 保存当前帧 (带检测框) 到 snapshots/
    SPACE   : 暂停 / 继续
"""
import os
import re
import glob
import time
import argparse

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import model.detector as detector_mod
import utils.utils
import utils.datasets


# 类别固定配色 (BGR), 与 data/dms-kaggle.names 对应
CLASS_COLORS = [
    (0, 255, 0),     # Open Eye    - 绿
    (0, 0, 255),     # Closed Eye  - 红
    (0, 165, 255),   # Cigarette   - 橙
    (255, 0, 255),   # Phone       - 品红
    (255, 200, 0),   # Seatbelt    - 青蓝
]

# 类别中英映射 (按 .names 文件中的顺序)
EN_TO_ZH = {
    "Open Eye":   "睁眼",
    "Closed Eye": "闭眼",
    "Cigarette":  "香烟",
    "Phone":      "手机",
    "Seatbelt":   "安全带",
}

# 备选中文字体路径 (Windows 常见)
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",     # 微软雅黑
    "C:/Windows/Fonts/msyhbd.ttc",   # 微软雅黑 Bold
    "C:/Windows/Fonts/simhei.ttf",   # 黑体
    "C:/Windows/Fonts/simsun.ttc",   # 宋体
]


# -------------------- 工具函数 --------------------

def find_best_weight(weights_dir: str):
    """从 weights_dir 中按文件名解析 mAP，返回 mAP 最高的 .pth 路径。"""
    pths = glob.glob(os.path.join(weights_dir, "*.pth"))
    if not pths:
        raise FileNotFoundError(f"在 {weights_dir} 下未找到 .pth 权重文件")
    best_path, best_ap = None, -1.0
    pat = re.compile(r"epoch-([0-9.]+)ap")
    for p in pths:
        m = pat.search(os.path.basename(p))
        if not m:
            continue
        try:
            ap = float(m.group(1))
        except ValueError:
            continue
        if ap > best_ap:
            best_ap, best_path = ap, p
    if best_path is None:
        best_path = max(pths, key=os.path.getmtime)
        best_ap = float("nan")
    return best_path, best_ap


def load_label_names(names_path: str):
    with open(names_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_chinese_font(size: int, font_path: str = None):
    candidates = [font_path] if font_path else []
    candidates += FONT_CANDIDATES
    for p in candidates:
        if p and os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    print("[demo_camera] 警告: 未找到中文字体，将使用 PIL 默认字体 (中文可能无法正常显示)")
    return ImageFont.load_default()


# letterbox 复用 utils.datasets.letterbox, 避免重复实现
letterbox = utils.datasets.letterbox


def map_box_to_orig(box, ratio, pad_l, pad_t, ori_w, ori_h):
    """letterbox 反变换: 模型输出坐标 -> 原图坐标, 并裁剪到画面内。"""
    x1 = (box[0] - pad_l) / ratio
    y1 = (box[1] - pad_t) / ratio
    x2 = (box[2] - pad_l) / ratio
    y2 = (box[3] - pad_t) / ratio
    x1 = int(max(0, min(ori_w - 1, x1)))
    y1 = int(max(0, min(ori_h - 1, y1)))
    x2 = int(max(0, min(ori_w - 1, x2)))
    y2 = int(max(0, min(ori_h - 1, y2)))
    return x1, y1, x2, y2


def preprocess(frame, width, height, device, keep_ratio=False):
    """stretch(默认) 或 letterbox 后转为 (1,3,H,W) float32 tensor / 255。"""
    if keep_ratio:
        canvas, ratio, pad_l, pad_t = letterbox(frame, width, height)
    else:
        canvas = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        h, w = frame.shape[:2]
        # 伪装成 letterbox 输出以重用反变换: ratio 不均匀时取两轴各自比例
        # 为保持接口一致性, 返回 None 表示调用方需使用 stretch 反变换
        ratio, pad_l, pad_t = None, 0, 0
    img = canvas.transpose(2, 0, 1)[None]
    tensor = torch.from_numpy(img).to(device).float() / 255.0
    return tensor, ratio, pad_l, pad_t


def warmup(net, cfg, device, n=2):
    """CUDA 预热: 跑 n 次空 forward, 抹平首帧延迟。"""
    dummy = torch.zeros(1, 3, cfg["height"], cfg["width"], device=device)
    with torch.no_grad():
        for _ in range(n):
            _ = net(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()


# -------------------- 绘制 --------------------

def draw_boxes_cv(frame, detections, class_colors):
    """detections: list of dict {x1,y1,x2,y2,score,cls_id,name_zh}. 仅画矩形, 不画文字。"""
    for d in detections:
        color = class_colors[d["cls_id"] % len(class_colors)]
        cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, 2)


def draw_texts_pil(frame_bgr, detections, hud_lines, font, font_small, class_colors):
    """
    一次性把所有中文文字渲染到 frame 上 (转成 PIL 画完再转回 BGR)。
    包括: 检测框上方的标签 + 左上角 HUD + 底部提示。
    """
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # --- 检测框标签 ---
    for d in detections:
        color_bgr = class_colors[d["cls_id"] % len(class_colors)]
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        text = f"{d['name_zh']} {d['score']:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = d["x1"], d["y1"] - th - 6
        if y < 0:  # 顶到画面外则放到框内
            y = d["y1"] + 2
        # 背景色块
        draw.rectangle([x, y, x + tw + 6, y + th + 6], fill=color_rgb)
        # 黑字
        draw.text((x + 3, y + 2), text, font=font, fill=(0, 0, 0))

    # --- 左上 HUD ---
    y = 8
    for line in hud_lines:
        # 黑色描边 (4 方向偏移) + 白字, 任意背景下都清晰
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((10 + dx, y + dy), line, font=font, fill=(0, 0, 0))
        draw.text((10, y), line, font=font, fill=(255, 255, 255))
        bbox = draw.textbbox((0, 0), line, font=font)
        y += (bbox[3] - bbox[1]) + 6

    # --- 底部提示 ---
    hint = "Q/ESC 退出   S 截图   SPACE 暂停/继续"
    bbox = draw.textbbox((0, 0), hint, font=font_small)
    th = bbox[3] - bbox[1]
    h = img_pil.size[1]
    yy = h - th - 10
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((10 + dx, yy + dy), hint, font=font_small, fill=(0, 0, 0))
    draw.text((10, yy), hint, font=font_small, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# -------------------- 主流程 --------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Yolo-FastestV2 摄像头实时推理 (中文显示)")
    parser.add_argument("--data", type=str, default="data/dms-kaggle.data",
                        help="训练配置 *.data 路径")
    parser.add_argument("--weights", type=str, default="",
                        help="权重 .pth 路径; 留空则自动选择 weights/ 下 mAP 最高的")
    parser.add_argument("--weights-dir", type=str, default="weights",
                        help="自动挑选权重时使用的目录")
    parser.add_argument("--cam", default="0",
                        help="摄像头索引 (整数) 或视频文件路径; 默认 0")
    parser.add_argument("--conf", type=float, default=0.6,
                        help="置信度阈值 (默认 0.6)")
    parser.add_argument("--iou", type=float, default=0.4,
                        help="NMS IoU 阈值")
    parser.add_argument("--width", type=int, default=0,
                        help="摄像头采集宽度 (0 = 默认)")
    parser.add_argument("--height", type=int, default=0,
                        help="摄像头采集高度 (0 = 默认)")
    parser.add_argument("--keep-ratio", dest="keep_ratio", action="store_true",
                        default=False, help="预处理使用 letterbox 等比缩放 (默认关)")
    parser.add_argument("--no-keep-ratio", dest="keep_ratio", action="store_false",
                        help="预处理直接 stretch 拉伸 (默认)")
    parser.add_argument("--font", type=str, default="",
                        help="中文字体文件路径 (.ttf/.ttc); 默认自动找微软雅黑/黑体/宋体")
    parser.add_argument("--font-size", type=int, default=22,
                        help="检测框/HUD 文字字号")
    parser.add_argument("--snapshot-dir", type=str, default="snapshots",
                        help="按 S 截图的保存目录")
    return parser.parse_args()


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

    if opt.weights:
        weight_path = opt.weights
        weight_ap = float("nan")
        assert os.path.exists(weight_path), f"找不到权重: {weight_path}"
    else:
        weight_path, weight_ap = find_best_weight(opt.weights_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo_camera] device       = {device}")
    print(f"[demo_camera] data config  = {opt.data}")
    print(f"[demo_camera] weight       = {weight_path} (mAP={weight_ap:.4f})")
    mode_str = 'letterbox' if opt.keep_ratio else 'stretch'
    print(f"[demo_camera] input size   = {cfg['width']} x {cfg['height']} ({mode_str})")
    print(f"[demo_camera] conf / iou   = {opt.conf} / {opt.iou}")

    # 模型加载 (weights_only=True 更安全 + 静音 FutureWarning)
    net = detector_mod.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    try:
        state = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weight_path, map_location=device)
    # 兼容 train.py 新格式 {epoch, state_dict, mAP, ...}
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    net.load_state_dict(state)
    net.eval()

    # 标签
    en_names = load_label_names(cfg["names"])
    zh_names = [EN_TO_ZH.get(n, n) for n in en_names]
    print(f"[demo_camera] classes      = {list(zip(en_names, zh_names))}")

    # 字体
    font = get_chinese_font(opt.font_size, opt.font or None)
    font_small = get_chinese_font(max(14, opt.font_size - 6), opt.font or None)

    # 摄像头 / 视频
    cap = open_capture(opt.cam, opt.width, opt.height)

    os.makedirs(opt.snapshot_dir, exist_ok=True)

    # 预热
    print("[demo_camera] CUDA / 模型预热中...")
    warmup(net, cfg, device, n=3)

    win_name = "Yolo-FastestV2 Camera Demo (CN)"
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
                        print("[demo_camera] 连续读帧失败 / 视频结束，退出")
                        break
                    print(f"[demo_camera] 读取帧失败，重试 ({fail_count}/3)...")
                    time.sleep(0.05)
                    continue
                fail_count = 0

                ori_h, ori_w = frame.shape[:2]

                # --- 预处理 (letterbox 或 stretch) ---
                tensor, ratio, pad_l, pad_t = preprocess(
                    frame, cfg["width"], cfg["height"], device,
                    keep_ratio=opt.keep_ratio,
                )

                # --- 推理 + 后处理 ---
                t0 = time.perf_counter()
                with torch.no_grad():
                    preds = net(tensor)
                    output = utils.utils.handel_preds(preds, cfg, device)
                    output_boxes = utils.utils.non_max_suppression(
                        output, conf_thres=opt.conf, iou_thres=opt.iou
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                last_infer_ms = (t1 - t0) * 1000.0

                # --- 反变换 + 类别索引保护 ---
                detections = []
                if len(output_boxes) > 0 and len(output_boxes[0]) > 0:
                    for b in output_boxes[0]:
                        b = b.tolist()
                        cls_id = int(b[5])
                        if cls_id < 0 or cls_id >= len(en_names):
                            continue
                        if opt.keep_ratio:
                            x1, y1, x2, y2 = map_box_to_orig(
                                b, ratio, pad_l, pad_t, ori_w, ori_h
                            )
                        else:
                            sw = ori_w / cfg["width"]
                            sh = ori_h / cfg["height"]
                            x1 = int(max(0, min(ori_w - 1, b[0] * sw)))
                            y1 = int(max(0, min(ori_h - 1, b[1] * sh)))
                            x2 = int(max(0, min(ori_w - 1, b[2] * sw)))
                            y2 = int(max(0, min(ori_h - 1, b[3] * sh)))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        detections.append({
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "score": float(b[4]),
                            "cls_id": cls_id,
                            "name_zh": zh_names[cls_id],
                            "name_en": en_names[cls_id],
                        })
                last_dets = detections

                # --- 绘制 ---
                draw_boxes_cv(frame, detections, CLASS_COLORS)
                last_render = frame  # 暂存原始 (含矩形, 无中文) 帧

            # FPS 平滑
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = inst_fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * inst_fps

            if last_render is None:
                continue

            # HUD 文本 (中文)
            hud_lines = [
                f"帧率 FPS: {fps_smooth:5.1f}",
                f"推理耗时: {last_infer_ms:5.1f} ms",
                f"检测数量: {len(last_dets)}",
            ]
            if paused:
                hud_lines.append("⏸ 已暂停")

            display = draw_texts_pil(
                last_render.copy() if paused else last_render,
                last_dets, hud_lines, font, font_small, CLASS_COLORS,
            )

            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(opt.snapshot_dir, f"snapshot_{ts}.png")
                cv2.imwrite(save_path, display)
                print(f"[demo_camera] 已保存截图 -> {save_path}")
            elif key == ord(" "):
                paused = not paused
                print(f"[demo_camera] {'暂停' if paused else '继续'}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
