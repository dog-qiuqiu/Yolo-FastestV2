import os
import argparse
import json
import numpy as np
from genanchors import gen_anchor

def main(args):
    config_path = args.config
    d_type = args.type

    with open(config_path) as f:
        config = json.load(f)

    time_interval = config['cqt']['hop'] / config['cqt']['sr']
    pitch_interval = config['cqt']['bins_per_octave'] / 12
    overlap_ratio = config['cqt']['overlap_ratio']
    duration = config['cqt']['duration']
    width_in_cfg_file = config['m_config']['width']
    height_in_cfg_file = config['m_config']['height']
    width = int(config['cqt']['duration'] / time_interval)
    height = config['cqt']['n_bins']
    min_duration = config['min_duration']
    hop = duration * (1 - overlap_ratio)
    subset_dir = os.path.join(config['dataset_dir'], d_type)
    filepaths = [os.path.join(config['dataset_dir'], 'label', filename.split('.')[0] + '.txt')
                 for filename in os.listdir(subset_dir)]

    boxs = []
    for filepath in filepaths:
        label = np.loadtxt(filepath, dtype='float32').reshape(-1, 3)
        length = label[-1, 1]
        num = int(np.ceil(length / hop))
        for onset, offset, pitch in label:
            h = config['cqt']['n_bins'] - (pitch - 21) * pitch_interval
            for i in range(num):
                start_t = - duration / 2 + i * hop
                stop_t = start_t + duration
                if onset < stop_t and offset > start_t:
                    t = (min(offset, stop_t) - max(onset, start_t))
                    if t > min_duration:
                        w = t / time_interval
                        boxs.append([w / width, h / height])
    boxs = np.array(boxs, dtype='float32')
    gen_anchor('./', 6, boxs, width_in_cfg_file, height_in_cfg_file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='the config file path')
    parser.add_argument('--type', '-t', default='train', help='the subset type')
    return parser.parse_args()

'''
计算训练集中的框
'''
if __name__ == '__main__':
    main(parse_args())