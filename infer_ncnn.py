"""NCNN 推理脚本: 用 python ncnn 跑 yolo-fastestv2-dms 模型.

输入:
  --param  .ncnn.param 网络结构
  --bin    .ncnn.bin   权重 (FP16)
输出形状与 ONNX 一致, 仅去掉 batch 维:
  out0: (22, 22, 20)
  out1: (11, 11, 20)
通道布局同 detector.export_onnx 分支:
  [0:12]  per-anchor reg (cx,cy,w,h) x 3 anchors, 已 sigmoid
  [12:15] per-anchor obj (1) x 3 anchors,        已 sigmoid
  [15:20] shared cls (5 classes),                 已 softmax
"""
import argparse
import os
import time

import cv2
import numpy as np
import ncnn

import utils.utils
from infer_onnx import decode, nms, CLASS_COLORS, preprocess


def parse_args():
    p = argparse.ArgumentParser(description="NCNN 推理 (yolo-fastestv2-dms)")
    p.add_argument('--data', type=str, required=True, help='*.data 配置')
    p.add_argument('--param', type=str, required=True, help='.ncnn.param 路径')
    p.add_argument('--bin', type=str, required=True, help='.ncnn.bin 路径')
    p.add_argument('--img', type=str, required=True, help='测试图像')
    p.add_argument('--conf', type=float, default=0.3, help='置信度阈值')
    p.add_argument('--iou', type=float, default=0.4, help='NMS IoU 阈值')
    p.add_argument('--out', type=str, default='infer_ncnn_result.png', help='输出图')
    p.add_argument('--threads', type=int, default=4, help='ncnn 线程数')
    p.add_argument('--bench', type=int, default=0, help='benchmark 次数')
    return p.parse_args()


def main():
    opt = parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    W, H = cfg['width'], cfg['height']
    NC = cfg['classes']
    NA = cfg['anchor_num']
    anchors = np.array(cfg['anchors']).reshape(2, NA, 2)

    print(f"[ncnn] param: {opt.param}")
    print(f"[ncnn] bin  : {opt.bin}")
    print(f"[ncnn] threads: {opt.threads}")

    # Class names
    LABELS = [f'cls{i}' for i in range(NC)]
    if cfg.get('names') and os.path.exists(cfg['names']):
        with open(cfg['names'], encoding='utf-8') as f:
            LABELS = [line.strip() for line in f if line.strip()]
        print(f"[ncnn] labels: {LABELS}")

    # Read image
    bgr = cv2.imread(opt.img)
    if bgr is None:
        raise FileNotFoundError(opt.img)
    ori_h, ori_w = bgr.shape[:2]
    print(f"[ncnn] image: {ori_w}x{ori_h}, model in: {W}x{H}")

    x = preprocess(bgr, W, H)            # (1, 3, H, W) NCHW float32 /255
    # 关键 1: cv2.transpose 后数组非连续, 必须 ascontiguousarray, 否则 ncnn 读乱
    # 关键 2: 必须保持 numpy 数组的引用, 否则 ncnn.Mat 内部指针会成野指针 -> segfault
    x_np = np.ascontiguousarray(x[0])

    with ncnn.Net() as net:
        net.opt.use_vulkan_compute = False
        net.load_param(opt.param)
        net.load_model(opt.bin)

        def run_once():
            x_mat = ncnn.Mat(x_np).clone()  # 每次新建 Mat, 避免跨 extractor 复用
            with net.create_extractor() as ex:
                ex.input('in0', x_mat)
                _, out0 = ex.extract('out0')
                _, out1 = ex.extract('out1')
                # extract 返回的 Mat 引用 ex 内的内存, 必须先复制再退出 with
                a0 = np.array(out0)
                a1 = np.array(out1)
            return a0, a1

        # warmup + forward
        run_once()
        t0 = time.perf_counter()
        out0, out1 = run_once()
        dt_fwd = (time.perf_counter() - t0) * 1000
        print(f"[ncnn] outputs: {out0.shape}, {out1.shape}")
        print(f"[ncnn] forward latency: {dt_fwd:.2f} ms")

        if opt.bench > 0:
            t0 = time.perf_counter()
            for _ in range(opt.bench):
                run_once()
            avg = (time.perf_counter() - t0) / opt.bench * 1000
            print(f"[bench] avg over {opt.bench}: {avg:.2f} ms ({1000/avg:.1f} FPS)")

    # decode 期望 (1, gh, gw, 20), 加回 batch 维
    outs = [out0[None], out1[None]]
    dets_all = []
    for i, out in enumerate(outs):
        gh = out.shape[1]
        stride = H / gh
        d = decode(out, anchors[i], stride, NC, NA, conf_thres=opt.conf)
        dets_all.append(d)
    dets = np.concatenate(dets_all, axis=0) if dets_all else np.zeros((0, 6))
    print(f"[ncnn] raw dets: {len(dets)}")
    dets = nms(dets, opt.iou)
    print(f"[ncnn] after NMS: {len(dets)}")

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
    print(f"[ncnn] saved -> {opt.out}")


if __name__ == '__main__':
    main()
