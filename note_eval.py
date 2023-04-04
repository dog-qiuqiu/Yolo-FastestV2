import mir_eval
import numpy as np
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--label', '-l', type=str)
parser.add_argument('--result', '-r', type=str)
parser.add_argument('--onset-win', type=float, default=0.05)
parser.add_argument('--offset', action='store_true', default=False)
parser.add_argument('--offset-win', type=float, default=0.05)
parser.add_argument('--save_json', type=str, default=None)

args = parser.parse_args()

def get_onset_or_offset(filepath, flag):
    times = []
    with open(filepath) as f:
        for line in f:
            onset, offset, _ = line.strip().split()
            if float(onset)<0:
                onset = 0
                print('nagative onset', filepath)
            if float(onset)>float(offset):
                return 'error'
                # raise ValueError(filepath, onset, offset)
            if float(offset)-float(onset)<1e-6:
                continue
            if flag=='onset':
                times.append(float(onset))
            elif flag=='offset':
                times.append([float(onset), float(offset)])
    if flag=='onset':
        times.sort()
    elif flag=='offset':
        times.sort(key=lambda x:x[0])
    return np.array(times)

if __name__=='__main__':
    labeldir = args.label
    resultdir = args.result
    offset = args.offset
    save_json = args.save_json
    filenames = [filename for filename in os.listdir(resultdir)] # if not filename.startswith('amale') ]
    filenames.sort()
    mean_f, mean_p, mean_r = 0, 0, 0
    mean_f1, mean_p1, mean_r1 = 0, 0, 0
    mean_f2, mean_p2, mean_r2 = 0, 0, 0

    cnt=0
    bads=[]
    if save_json:
        onset_ps, onset_rs, onset_fs = [], [], []
        offset_ps, offset_rs, offset_fs = [], [], []

    for filename in filenames:
        labelpath = os.path.join(labeldir, filename)
        resultpath = os.path.join(resultdir, filename)

        ref_onsets = get_onset_or_offset(labelpath, 'onset')
        est_onsets = get_onset_or_offset(resultpath, 'onset')
        if est_onsets == 'error':
            continue
        if est_onsets.shape[0]>0:
            f, p, r=mir_eval.onset.f_measure(ref_onsets, est_onsets, window=args.onset_win)
        else:
            f, p, r=0, 0, 0
        mean_f += f
        mean_p += p
        mean_r += r
        if save_json:
            onset_ps.append(p)
            onset_rs.append(r)
            onset_fs.append(f)

        if f<0.8:
            cnt+=1
            bads.append([filename, f])
        print(filename)
        print('f: ', f, 'p: ', p, 'r: ', r)

        if offset:
            ref_intervals = get_onset_or_offset(labelpath, 'offset')
            est_intervals = get_onset_or_offset(resultpath, 'offset')

            if est_intervals.shape[0]>0:
                p1, r1, f1 = mir_eval.transcription.offset_precision_recall_f1(
                    ref_intervals, est_intervals, offset_ratio=0.2, offset_min_tolerance=args.offset_win)
            else:
                p1, r1, f1 = 0, 0, 0
            mean_f1 += f1
            mean_p1 += p1
            mean_r1 += r1
            print('f1: ', f1, 'p1: ', p1, 'r1: ', r1)

            if save_json:
                offset_ps.append(p1)
                offset_rs.append(r1)
                offset_fs.append(f1)

    mean_f /= len(filenames)
    mean_p /= len(filenames)
    mean_r /= len(filenames)

    print('mean_f: %.2f mean_p: %.2f mean_r: %.2f' % (100 * mean_f, 100 * mean_p, 100 * mean_r))

    if offset:
        mean_f1 /= len(filenames)
        mean_p1 /= len(filenames)
        mean_r1 /= len(filenames)
        print('mean_f1: %.2f mean_p1: %.2f mean_r1: %.2f'%(100*mean_f1, 100*mean_p1, 100*mean_r1))

    if save_json:
        onset_fs = [round(value*100, 2) for value in onset_fs]
        onset_ps = [round(value*100, 2) for value in onset_ps]
        onset_rs = [round(value*100, 2) for value in onset_rs]
        data = dict()
        for i in range(len(filenames)):
            data[filenames[i]] = [onset_fs[i], onset_ps[i], onset_rs[i]]
        if offset:
            offset_fs = [round(value*100, 2) for value in offset_fs]
            offset_ps = [round(value*100, 2) for value in offset_ps]
            offset_rs = [round(value*100, 2) for value in offset_rs]
            for i in range(len(filenames)):
                data[filenames[i]].extend([onset_fs[i], onset_ps[i], onset_rs[i]])
        data = json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '))
        save_path = save_json+'_win_%.2f.json'%args.onset_win
        with open(save_path, 'w') as f: f.write(data)