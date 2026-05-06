"""ONNX 推理脚本: 用 onnxruntime 跑 yolo-fastestv2-dms 模型.

输出形状: detector(export_onnx=True) 已做 sigmoid/softmax, 输出 NHWC=20:
  [0:12]  per-anchor reg  (cx,cy,w,h) x 3 anchors, 已 sigmoid
  [12:15] per-anchor obj  (1) x 3 anchors,        已 sigmoid
  [15:20] shared cls      (5 classes),            已 softmax
"""
import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

import utils.utils


CLASS_COLORS = [
    (0, 255, 0),      # eye_open / 0
    (255, 128, 0),    # eye_closed / 1 (蓝在 BGR)
    (0, 200, 255),    # mouth_open / 2
    (255, 0, 255),    # mouth_closed / 3
    (0, 0, 255),      # phone? / 4
]


def parse_args():
    p = argparse.ArgumentParser(description="ONNX 推理 (yolo-fastestv2-dms)")
    p.add_argument('--data', type=str, required=True, help='*.data 配置')
    p.add_argument('--onnx', type=str, required=True, help='.onnx 模型路径')
    p.add_argument('--img', type=str, required=True, help='测试图像')
    p.add_argument('--conf', type=float, default=0.3, help='置信度阈值')
    p.add_argument('--iou', type=float, default=0.4, help='NMS IoU 阈值')
    p.add_argument('--out', type=str, default='infer_onnx_result.png', help='输出图')
    p.add_argument('--provider', type=str, default='auto',
                   help="onnxruntime provider: 'auto' | 'cpu' | 'cuda'")
    p.add_argument('--bench', type=int, default=0,
                   help='重复推理 N 次做 benchmark (0=不 bench)')
    return p.parse_args()


def select_provider(name):
    avail = ort.get_available_providers()
    if name == 'cpu':
        return ['CPUExecutionProvider']
    if name == 'cuda' and 'CUDAExecutionProvider' in avail:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if name == 'auto':
        return [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in avail]
    return ['CPUExecutionProvider']


def preprocess(bgr, w, h):
    """stretch resize -> NCHW float32 (/255)."""
    canvas = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return x


def make_grid_np(gh, gw, na):
    yv, xv = np.meshgrid(np.arange(gh), np.arange(gw), indexing='ij')
    return np.stack((xv, yv), axis=-1)[..., None, :].repeat(na, axis=-2)  # (gh,gw,na,2)


def decode(out, anchors_layer, stride, num_cls=5, na=3, conf_thres=0.05):
    """ONNX 输出 (1, gh, gw, 20) -> list of [x1,y1,x2,y2,score,cls] (输入空间像素).

    out[..., 0:12]  per-anchor reg (sigmoid 过)
    out[..., 12:15] per-anchor obj (sigmoid 过)
    out[..., 15:20] shared cls (softmax 过)
    """
    o = out[0]  # (gh, gw, 20)
    gh, gw, _ = o.shape
    reg = o[..., 0:12].reshape(gh, gw, na, 4)   # (gh,gw,na,4)
    obj = o[..., 12:15].reshape(gh, gw, na, 1)  # (gh,gw,na,1)
    cls = o[..., 15:15 + num_cls]                # (gh,gw,5) shared

    # cx, cy
    grid = make_grid_np(gh, gw, na).astype(np.float32)  # (gh,gw,na,2)
    cxy = (reg[..., 0:2] * 2.0 - 0.5 + grid) * stride

    # w, h
    anchors_layer = anchors_layer.astype(np.float32)  # (na, 2)
    wh = (reg[..., 2:4] * 2.0) ** 2 * anchors_layer  # broadcast (gh,gw,na,2)

    # score per class = obj * cls
    cls_b = cls[:, :, None, :].repeat(na, axis=2)            # (gh,gw,na,5)
    score_all = obj * cls_b                                  # (gh,gw,na,5)

    cls_id = np.argmax(score_all, axis=-1)                   # (gh,gw,na)
    cls_score = np.max(score_all, axis=-1)                   # (gh,gw,na)

    mask = cls_score > conf_thres
    cxy = cxy[mask]; wh = wh[mask]
    cls_id = cls_id[mask]; cls_score = cls_score[mask]

    if cxy.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    x1 = cxy[:, 0] - wh[:, 0] / 2
    y1 = cxy[:, 1] - wh[:, 1] / 2
    x2 = cxy[:, 0] + wh[:, 0] / 2
    y2 = cxy[:, 1] + wh[:, 1] / 2
    return np.stack([x1, y1, x2, y2, cls_score, cls_id.astype(np.float32)], axis=1)


def nms(dets, iou_thres):
    """简单 per-class NMS, 输入 [x1,y1,x2,y2,score,cls]."""
    if len(dets) == 0:
        return dets
    out = []
    for c in np.unique(dets[:, 5]):
        sub = dets[dets[:, 5] == c]
        order = sub[:, 4].argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]; keep.append(sub[i])
            if len(order) == 1: break
            xx1 = np.maximum(sub[i, 0], sub[order[1:], 0])
            yy1 = np.maximum(sub[i, 1], sub[order[1:], 1])
            xx2 = np.minimum(sub[i, 2], sub[order[1:], 2])
            yy2 = np.minimum(sub[i, 3], sub[order[1:], 3])
            inter = np.maximum(0., xx2 - xx1) * np.maximum(0., yy2 - yy1)
            area_i = (sub[i, 2] - sub[i, 0]) * (sub[i, 3] - sub[i, 1])
            area_o = (sub[order[1:], 2] - sub[order[1:], 0]) * (sub[order[1:], 3] - sub[order[1:], 1])
            iou = inter / (area_i + area_o - inter + 1e-7)
            order = order[1:][iou < iou_thres]
        out.extend(keep)
    return np.array(out)


def main():
    opt = parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    W, H = cfg['width'], cfg['height']
    NC = cfg['classes']
    NA = cfg['anchor_num']
    anchors = np.array(cfg['anchors']).reshape(2, NA, 2)  # 2 个尺度

    # Load
    providers = select_provider(opt.provider)
    print(f"[onnx] providers: {providers}")
    sess = ort.InferenceSession(opt.onnx, providers=providers)
    in_name = sess.get_inputs()[0].name

    # Class names
    names_path = cfg.get('names', '')
    LABELS = [f'cls{i}' for i in range(NC)]
    if names_path and os.path.exists(names_path):
        with open(names_path, encoding='utf-8') as f:
            LABELS = [line.strip() for line in f if line.strip()]
        print(f"[onnx] labels: {LABELS}")

    # Read image
    bgr = cv2.imread(opt.img)
    if bgr is None:
        raise FileNotFoundError(opt.img)
    ori_h, ori_w = bgr.shape[:2]
    print(f"[onnx] image: {ori_w}x{ori_h}, model in: {W}x{H}")

    x = preprocess(bgr, W, H)

    # Forward
    t0 = time.perf_counter()
    outs = sess.run(None, {in_name: x})
    dt_fwd = (time.perf_counter() - t0) * 1000
    print(f"[onnx] outputs: {[o.shape for o in outs]}")
    print(f"[onnx] forward latency: {dt_fwd:.2f} ms")

    if opt.bench > 0:
        t0 = time.perf_counter()
        for _ in range(opt.bench):
            sess.run(None, {in_name: x})
        avg = (time.perf_counter() - t0) / opt.bench * 1000
        print(f"[bench] avg over {opt.bench}: {avg:.2f} ms ({1000/avg:.1f} FPS)")

    # 后处理: out[0] is 22x22 (stride 16), out[1] is 11x11 (stride 32)
    dets_all = []
    for i, out in enumerate(outs):
        gh, gw = out.shape[1:3]
        stride = H / gh
        d = decode(out, anchors[i], stride, NC, NA, conf_thres=opt.conf)
        dets_all.append(d)
    dets = np.concatenate(dets_all, axis=0)
    print(f"[onnx] raw dets: {len(dets)}")
    dets = nms(dets, opt.iou)
    print(f"[onnx] after NMS: {len(dets)}")

    # 反映射到原图
    sx = ori_w / W; sy = ori_h / H
    canvas = bgr.copy()
    for d in dets:
        x1, y1, x2, y2, sc, c = d
        x1 = int(max(0, x1 * sx)); y1 = int(max(0, y1 * sy))
        x2 = int(min(ori_w - 1, x2 * sx)); y2 = int(min(ori_h - 1, y2 * sy))
        cls_id = int(c)
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{LABELS[cls_id] if cls_id < len(LABELS) else cls_id} {sc:.2f}"
        cv2.putText(canvas, text, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"  {LABELS[cls_id] if cls_id < len(LABELS) else cls_id}: "
              f"({x1},{y1})-({x2},{y2}) score={sc:.3f}")

    cv2.imwrite(opt.out, canvas)
    print(f"[onnx] saved -> {opt.out}")


if __name__ == '__main__':
    main()
