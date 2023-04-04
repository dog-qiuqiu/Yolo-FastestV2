import os
import argparse
import torch
import model.detector
import utils.utils
import utils.datasets
import json

def write(savepath, notes):
    with open(savepath, 'w') as f:
        for onset, offset, pitch, _ in notes:
            f.write('{:.3f}\t{:.3f}\t{:.3f}\n'.format(onset, offset, pitch))

if __name__ == '__main__':
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='', 
                        help='Specify config')
    parser.add_argument('--weights', '-w', type=str, default='', 
                        help='The path of the .pth model to be transformed')
    parser.add_argument('--testdir','-t', type=str, default='', 
                        help='The path of test data')

    args = parser.parse_args()
    assert os.path.exists(args.weights), "请指定正确的模型路径"

    #模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = json.load(f)

    net = model.detector.Detector(
        cfg["cqt"], cfg["m_config"]["width"], cfg["m_config"]["height"], 
        cfg["m_config"]["anchor_num"], False, 
        convert2image=cfg["m_config"]["convert2image"]).to(device)
    net.load_state_dict(torch.load(args.weights, map_location=device))
    cqt_transform = model.detector.CQTSpectrogram(cfg["cqt"], cfg["m_config"]["width"], cfg["m_config"]["height"], interpolate=False)

    #sets the module in eval node
    net.eval()
    batch_size = int(cfg["opt"]["batch_size"] / cfg["opt"]["subdivisions"])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    batch_size = 128
    filepaths = [os.path.join(args.testdir, file) for file in os.listdir(args.testdir)]

    dumpdir = cfg["test"]["test_checkdata_dir"]
    note_save_dir = os.path.join(dumpdir, 'res')
    os.makedirs(dumpdir, exist_ok=True)
    os.makedirs(note_save_dir, exist_ok=True)

    for fileno, filepath in enumerate(filepaths):
        cfg["cqt"]["overlap_ratio"] = cfg["test"]["overlap_ratio"]
        testset = utils.datasets.TestDataset(cfg["cqt"], filepath)
        loader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw,
                                             drop_last=False,
                                             persistent_workers=True
                                             )
        output_boxes = []
        for index, data in enumerate(loader): # [6, 80448] 等效于多张频谱图
            data = data.to(device)
            preds = net(data)
            cqt = preds[-1]
            preds = preds[: -1]

            # 把预测出来的归一化坐标转换到yolox输入图像的尺寸
            output = utils.utils.handel_preds(preds, cfg, device)
            # 可以当做一个二维list，第一层代表第几个数据，第二层为每张图像上框的数量
            output_box = utils.utils.non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)
            output_boxes.extend(output_box)

        scale_h = cfg["cqt"]["n_bins"] / (cfg["cqt"]["bins_per_octave"] / 12) / cfg["m_config"]["height"]
        scale_w = cfg["cqt"]["duration"] / cfg["m_config"]["width"]
        hop = (1. - cfg["cqt"]["overlap_ratio"]) * cfg["m_config"]["width"]
        total_notes = utils.utils.convert_boxs_to_notes(output_boxes, hop, scale_h, scale_w, cfg["m_config"]["width"])
        total_notes.sort(key=lambda x: x[0])
        print(filepath)
        feature = cqt_transform(testset.get_total_audio())[0, 0].numpy()
        basename = os.path.basename(filepath).split('.')[0]
        write(os.path.join(note_save_dir, basename + '.txt'), total_notes)

        if dumpdir:
            dumppath = os.path.join(dumpdir, basename + '.png')
            utils.utils.dump_test_data(feature, dumppath, total_notes, cfg["cqt"], no=str(fileno))
            # exit(0)