"""PyTorch -> ONNX 导出.

支持两种权重格式:
  - 纯 state_dict (旧仓库格式: dms-kaggle-150-epoch-0.699052ap-model.pth)
  - {epoch, state_dict, ...} dict 格式 (新 train.py 输出: dms-kaggle-best.pth)
"""
import argparse
import os

import torch
import onnx

import utils.utils
import model.detector as detector_mod   # 避免与下面变量名冲突


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch -> ONNX 导出")
    parser.add_argument('--data', type=str, required=True,
                        help='训练配置 *.data 路径')
    parser.add_argument('--weights', type=str, required=True,
                        help='待导出的 .pth 权重路径')
    parser.add_argument('--output', type=str, default='./model.onnx',
                        help='输出 .onnx 路径 (默认 ./model.onnx)')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset 版本 (默认 11, 与 ncnn 兼容性好)')
    parser.add_argument('--dynamic', action='store_true',
                        help='导出 dynamic batch axis (默认固定 batch=1, 兼容性更好)')
    parser.add_argument('--simplify', action='store_true',
                        help='导出后用 onnxsim 简化模型 (推荐, ncnn 转换更稳定)')
    parser.add_argument('--device', type=str, default='cpu',
                        help="导出设备: 'cpu'(默认, 推荐) | 'cuda'")
    return parser.parse_args()


def load_state_dict_compat(path, device):
    """同时兼容纯 state_dict 与 {epoch,state_dict,...} 字典."""
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and 'state_dict' in obj:
        info = ' '.join(
            f'{k}={obj[k]}' for k in ('epoch', 'mAP') if k in obj
        )
        print(f"[load] dict 格式 ({info}) -> 取 state_dict")
        return obj['state_dict']
    print("[load] 纯 state_dict 格式")
    return obj


def main():
    opt = parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    device = torch.device(opt.device)

    print("==== ONNX 导出配置 ====")
    print(f"  weights : {opt.weights}")
    print(f"  output  : {opt.output}")
    print(f"  opset   : {opt.opset}")
    print(f"  dynamic : {opt.dynamic}")
    print(f"  simplify: {opt.simplify}")
    print(f"  device  : {device}")
    print(f"  input   : 1x3x{cfg['height']}x{cfg['width']}")
    print(f"  classes : {cfg['classes']}")
    print(f"  anchors : {cfg['anchor_num']} per scale")

    # 模型: 第 4 个参数 export_onnx=True 让 detector 走 onnx 友好分支
    net = detector_mod.Detector(
        cfg["classes"], cfg["anchor_num"], True, True
    ).to(device)
    state = load_state_dict_compat(opt.weights, device)
    net.load_state_dict(state)
    net.eval()

    # 输入 / 动态轴
    dummy = torch.rand(1, 3, cfg["height"], cfg["width"]).to(device)
    dyn = {'input': {0: 'batch'}, 'output': {0: 'batch'}} if opt.dynamic else None

    print("\n==== 开始导出 ====")
    torch.onnx.export(
        net, dummy, opt.output,
        export_params=True,
        opset_version=opt.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dyn,
    )
    print(f"[done] 写入 {opt.output} ({os.path.getsize(opt.output) / 1024:.1f} KB)")

    # 校验 ONNX 结构
    m = onnx.load(opt.output)
    onnx.checker.check_model(m)
    print(f"[check] ONNX 结构合法, ir_version={m.ir_version}, opset={m.opset_import[0].version}")

    if opt.simplify:
        print("\n==== onnxsim 简化 ====")
        from onnxsim import simplify
        m_sim, ok = simplify(m)
        if not ok:
            print("[warn] onnxsim 简化失败, 保留原模型")
        else:
            sim_path = opt.output.replace('.onnx', '-sim.onnx')
            onnx.save(m_sim, sim_path)
            print(f"[done] 简化后 -> {sim_path} ({os.path.getsize(sim_path) / 1024:.1f} KB)")


if __name__ == '__main__':
    main()
