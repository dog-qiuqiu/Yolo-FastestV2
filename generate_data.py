import argparse
import os
import numpy as np
import soundfile as sf
from multiprocessing import Pool
from tqdm import tqdm
import librosa
import json

def worker(src_path, dest_path, target_sr):
    audio, sr = sf.read(src_path, dtype='float32')
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type='kaiser_fast')
    audio = (audio * 32768).astype('int16')
    np.save(dest_path, audio)

def preparse_subset(src_dir, dest_dir, target_sr):
    os.makedirs(dest_dir, exist_ok=True)
    pairs = []
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir, filename.split('.')[0] + '.npy')
        pairs.append([src_path, dest_path])

    worker(*pairs[0], target_sr)
    pbar = tqdm(desc='train data', total=len(pairs))
    pbar_update = lambda *args: pbar.update()
    pool = Pool(24)
    [pool.apply_async(worker, [*pair, target_sr], callback=pbar_update) for pair in pairs ]
    pool.close()
    pool.join() 
    
  
def main(args):
    config_path = args.config
    with open(config_path) as f:
        config = json.load(f)

    target_sr = config['cqt']['sr']
    dataset_dir = config['dataset_dir']
    save_dir = config['datadump_dir']

    print('preparing training data...')
    src_train_dir = os.path.join(dataset_dir, 'train')
    dest_train_dir = os.path.join(save_dir, 'train')
    preparse_subset(src_train_dir, dest_train_dir, target_sr)

    print('preparing evaluating data...')
    src_valid_dir = os.path.join(dataset_dir, 'valid')
    dest_valid_dir = os.path.join(save_dir, 'valid')
    preparse_subset(src_valid_dir, dest_valid_dir, target_sr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='the config file path')
    return parser.parse_args()

'''
主要实现的降采样，将原始信号采样频率统一变为16KHz,并存在numpy
'''
if __name__ == '__main__':
    main(parse_args())