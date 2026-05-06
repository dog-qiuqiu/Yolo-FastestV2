'''
Created on Feb 20, 2017

@author: jumabek
'''
from os import listdir
from os.path import isfile, join
import argparse
import cv2
import numpy as np
import sys
import os
import shutil
import random
import math

def IOU(x,centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w,c_h = centroid
        w,h = x
        if c_w>=w and c_h>=h:
            similarity = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape
    return np.array(similarities) 

def avg_IOU(X,centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        #note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum+= max(IOU(X[i],centroids)) 
    return sum/n

def write_anchors_to_file(centroids,X,anchor_file, width_in_cfg_file, height_in_cfg_file):
    f = open(anchor_file,'w')
    
    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0]*=width_in_cfg_file
        anchors[i][1]*=height_in_cfg_file
         

    widths = anchors[:,0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])
        
    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, '%(anchors[i,0],anchors[i,1]))

    #there should not be comma after last anchor, that's why
    last_idx = int(sorted_indices[-1])
    f.write('%0.2f,%0.2f\n'%(anchors[last_idx, 0], anchors[last_idx, 1]))
    
    f.write('%f\n'%(avg_IOU(X,centroids)))
    print()

def kmeans(X,centroids,eps,anchor_file, width_in_cfg_file, height_in_cfg_file):
    
    N = X.shape[0]
    iterations = 0
    k,dim = centroids.shape
    prev_assignments = np.ones(N)*(-1)    
    iter = 0
    old_D = np.zeros((N,k))

    while True:
        D = [] 
        iter+=1           
        for i in range(N):
            d = 1 - IOU(X[i],centroids)
            D.append(d)
        D = np.array(D) # D.shape = (N,k)
        
        print("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D))))
            
        #assign samples to centroids 
        assignments = np.argmin(D,axis=1)
        
        if (assignments == prev_assignments).all() :
            print("Centroids = ",centroids)
            write_anchors_to_file(centroids,X,anchor_file, width_in_cfg_file, height_in_cfg_file)
            return

        #calculate new centroids
        centroid_sums=np.zeros((k,dim), dtype=np.float64)
        for i in range(N):
            centroid_sums[assignments[i]]+=X[i]        
        for j in range(k):            
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j))
        
        prev_assignments = assignments.copy()     
        old_D = D.copy()  

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--traintxt', default = '', 
                        help='path to traintxt\n' )
    parser.add_argument('--output_dir', default = './', type = str, 
                        help='Output anchor directory\n' )  
    parser.add_argument('--num_clusters', default = 6, type = int, 
                        help='number of clusters\n' )  
    
    parser.add_argument('--input_width', default = 352, type = int, 
                        help='model input width\n' )  

    parser.add_argument('--input_height', default = 352, type = int, 
                        help='model input height\n' )  

    parser.add_argument('--keep-ratio', dest='keep_ratio', action='store_true',
                        default=False,
                        help='letterbox-aware 模式 (默认关): 按 letterbox ratio 将每个 label 变换到网络输入空间后聚类, 使 anchor 与训练目标在同一空间')
    parser.add_argument('--no-keep-ratio', dest='keep_ratio', action='store_false',
                        help='使用原始 YOLO 归一化尺寸聚类 (默认, stretch 训练路线使用)')

    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    f = open(args.traintxt)
  
    lines = [line.rstrip('\n') for line in f.readlines()]
    
    annotation_dims = []

    # keep_ratio 缓存: 相同尺寸的图只算一次 letterbox ratio
    size_cache = {}
    input_w = args.input_width
    input_h = args.input_height
    print(f'[genanchors] keep_ratio={args.keep_ratio}, input={input_w}x{input_h}')

    missing_imgs = 0
    for line in lines:
        img_path = line
        # 兼容多种数据集布局: VOC 风格 (JPEGImages -> labels) 或 YOLO 标准 (images -> labels)
        line_lbl = line.replace('JPEGImages', 'labels')
        line_lbl = line_lbl.replace(os.sep + 'images' + os.sep,
                                    os.sep + 'labels' + os.sep)
        line_lbl = line_lbl.replace('/images/', '/labels/')
        # 替换扩展名为 .txt (使用 splitext 避免文件名内含点导致截断)
        base, _ = os.path.splitext(line_lbl)
        line_lbl = base + '.txt'
        if not os.path.exists(line_lbl):
            print(f'[skip] 标签不存在: {line_lbl}')
            continue

        # letterbox-aware: 获取原图尺寸并计算 letterbox ratio
        if args.keep_ratio:
            if img_path in size_cache:
                ori_w, ori_h, ratio = size_cache[img_path]
            else:
                if not os.path.exists(img_path):
                    missing_imgs += 1
                    continue
                im = cv2.imread(img_path)
                if im is None:
                    missing_imgs += 1
                    continue
                ori_h, ori_w = im.shape[:2]
                ratio = min(input_w / ori_w, input_h / ori_h)
                size_cache[img_path] = (ori_w, ori_h, ratio)

        f2 = open(line_lbl)
        for ln in f2.readlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 5:
                continue
            w, h = float(parts[3]), float(parts[4])
            if args.keep_ratio:
                # YOLO 归一化 → 原图像素 → letterbox 像素 → 网络输入归一化
                w = w * ori_w * ratio / input_w
                h = h * ori_h * ratio / input_h
            annotation_dims.append((w, h))
        f2.close()

    if missing_imgs:
        print(f'[warn] {missing_imgs} 张图无法读取, 已跳过')
    annotation_dims = np.array(annotation_dims)
  
    eps = 0.005
    width_in_cfg_file = args.input_width
    height_in_cfg_file = args.input_height

    if args.num_clusters == 0:
        for num_clusters in range(1,11): #we make 1 through 10 clusters 
            anchor_file = join( args.output_dir,'anchors%d.txt'%(num_clusters))

            indices = [ random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
            centroids = annotation_dims[indices]
            kmeans(annotation_dims,centroids,eps,anchor_file, width_in_cfg_file, height_in_cfg_file)
            print('centroids.shape', centroids.shape)
    else:
        anchor_file = join( args.output_dir,'anchors%d.txt'%(args.num_clusters))
        indices = [ random.randrange(annotation_dims.shape[0]) for i in range(args.num_clusters)]
        centroids = annotation_dims[indices]
        kmeans(annotation_dims,centroids,eps,anchor_file, width_in_cfg_file, height_in_cfg_file)
        print('centroids.shape', centroids.shape)

if __name__=="__main__":
    main(sys.argv)
