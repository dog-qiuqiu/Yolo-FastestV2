import argparse
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

'''
分别统计音高和音符时长的分布
'''
def main(args):
    labeldir = args.label_dir
    pitches = [0 for _ in range(87)] # 每个音符的区间范围为[21 + i, 22 + i)
    min_duration = float('inf')
    max_duration = 0.
    for file in tqdm(os.listdir(labeldir)):
        labelpath = os.path.join(labeldir, file)
        label = np.loadtxt(labelpath).reshape(-1, 3)
        for onset, offset, pitch in label:
            pitch_index = int(pitch - 21)
            try:
              pitches[pitch_index] += 1
            except:
              pass
            duration = offset - onset
            min_duration = min(duration, min_duration)
            max_duration = max(duration, max_duration)
    
    # 绘制音高分布直方图
    indexs = [idx for idx in range(22, 109)]
    plt.bar(indexs, pitches)
    plt.savefig('pitch_distribution.png')
    plt.close()

    print('note duration: {:.3f} ~ {:.3f} s'.format(min_duration, max_duration))
    time_interval = 0.05 # 50ms为一个间隔
    min_count = int(min_duration / time_interval)
    lower_time = min_count * time_interval
    max_count = int(np.ceil(max_duration / time_interval))
    times = [0 for _ in range(max_count - min_count)]
    for file in tqdm(os.listdir(labeldir)):
        labelpath = os.path.join(labeldir, file)
        label = np.loadtxt(labelpath).reshape(-1, 3)
        for onset, offset, _ in label:
            duration = offset - onset
            time_index = int((duration - lower_time) // time_interval)
            times[time_index] += 1

    indexs = [idx * time_interval for idx in range(min_count, max_count)]
    plt.bar(indexs, times)
    plt.savefig('time_distribution.png')
    plt.close()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dir', help='the label dir path')
    return parser.parse_args()

'''
SSVD 2.0中的数据经过统计, 音高覆盖范围为40~80 音符时长0~3s
'''
if __name__ == '__main__':
    main(parse_args())