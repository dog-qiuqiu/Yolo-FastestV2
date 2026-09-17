import argparse

import onnx
import torch
import torch.nn as nn
import model.detector
import utils.utils
from model.backbone.shufflenetv2 import replace_shuffle_blocks_for_export


def fuse_conv_and_batch_norm(module):
    """Fuse adjacent Conv2d/BatchNorm2d pairs in an evaluation model."""
    for child in module.children():
        fuse_conv_and_batch_norm(child)

    if isinstance(module, nn.Sequential):
        layers = list(module.children())
        for index in range(len(layers) - 1):
            if isinstance(layers[index], nn.Conv2d) and isinstance(layers[index + 1], nn.BatchNorm2d):
                module[index] = torch.nn.utils.fuse_conv_bn_eval(
                    layers[index], layers[index + 1]
                )
                module[index + 1] = nn.Identity()
        # Remove BatchNorm placeholders so ONNX does not retain Identity nodes.
        module._modules = nn.Sequential(
            *(layer for layer in module.children() if not isinstance(layer, nn.Identity))
        )._modules


def remove_initializer_identity_nodes(onnx_path):
    """Remove exporter-created Identity aliases for shared constant weights."""
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    initializer_names = {initializer.name for initializer in graph.initializer}
    aliases = {
        node.output[0]: node.input[0]
        for node in graph.node
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1
        and node.input[0] in initializer_names
    }
    if not aliases:
        return

    for node in graph.node:
        for index, input_name in enumerate(node.input):
            while input_name in aliases:
                input_name = aliases[input_name]
            node.input[index] = input_name
    retained_nodes = [node for node in graph.node if node.output[0] not in aliases]
    del graph.node[:]
    graph.node.extend(retained_nodes)
    onnx.save(onnx_model, onnx_path)

if __name__ == '__main__':
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', 
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='', 
                        help='The path of the .pth model to be transformed')

    parser.add_argument('--output', type=str, default='./model.onnx', 
                        help='The path where the onnx model is saved')

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Export the native six raw NCHW branches; post-processing stays external.
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True, False).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    #sets the module in eval node
    model.eval()
    fuse_conv_and_batch_norm(model)
    replace_shuffle_blocks_for_export(model)

    test_data = torch.rand(1, 3, cfg["height"], cfg["width"], device=device)
    torch.onnx.export(model,                    #model being run
                     test_data,                 # model input (or a tuple for multiple inputs)
                     opt.output,               # where to save the model (can be a file or file-like object)
                     export_params=True,        # store the trained parameter weights inside the model file
                     opset_version=9,           # the ONNX version to export the model to
                     do_constant_folding=True,  # whether to execute constant folding for optimization
                     dynamo=False,               # use the legacy exporter to preserve opset 9
                     output_names=["reg_26", "obj_26", "cls_26", "reg_13", "obj_13", "cls_13"])
    remove_initializer_identity_nodes(opt.output)

#python pytorch2onnx.py --data data/coco.data --weights weights/best.pt --output wildlife_fused_convselect_opset9.onnx
    
