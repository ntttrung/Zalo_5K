from ntpath import join
import cv2
import glob
import pandas as pd
import os
import numpy as np  
import math
import itertools  


def compute_distance(cor1, cor2):
    return math.sqrt((cor1[0] - cor2[0])**2 + (cor1[1] - cor2[1])**2)

def compute_area(ls):
    return math.sqrt((ls[1]-ls[3])**2 * (ls[2] -ls[0])**2)

def ratio_area(ls1, ls2, max_area):
    s1 = compute_area(ls1)
    s2 = compute_area(ls2)
    if s1 > s2:
        return s1/s2
    return s2/s1

def calculateCentroid(ls):
    x_mid = (ls[0] + ls[2]) // 2
    y_mid = (ls[1] + ls[3]) // 2
    cor = []
    cor.extend([x_mid, y_mid])
    return cor

def real_distance(ls1, ls2, max_area):
    x1 = calculateCentroid(ls1)
    x2 = calculateCentroid(ls2)
    ratio = (ratio_area(ls1, ls2, max_area))
    return ratio**2 * (compute_distance(x1, x2)) 

def compute_distancing(label):
    df = pd.read_csv(r"C:\NTT\Zalo_5K\submission.csv")
    APS = 1300 * 1600
    source = r'C:\Users\trung\Downloads\AI_zalo_5K\public_test\images'
    list_path = glob.glob(label+ r'\*.txt')
    count = 0
    for path in list_path:
        name = os.path.basename(path)
        ind  = list_path.index(path)
        path_image = join(source, name.replace("txt","jpg"))
        img = cv2.imread(path_image)
        # img = cv2.resize(img, (800,800))
        dh, dw, _ = img.shape
        ra = math.sqrt(dh*dw/APS)
        fl = open(join(label, name), 'r')
        data = fl.readlines()
        fl.close()
        ls = []

        for dt in data:
            corr = []
            # Split string to float
            _, x, y, w, h = map(float, dt.split(' '))

            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2) * dh)
            
            if l < 0:
                l = 0
            if r > dw - 1:
                r = dw - 1
            if t < 0:
                t = 0
            if b > dh - 1:
                b = dh - 1
            corr.extend([l, t, r, b])
            ls.append(corr)
        list_area = []
        for i in ls:
            list_area.append(compute_area(i))
        max_area = max(list_area)
        permute = list(range(0, len(ls)))

        for i in itertools.permutations(permute, 2):
            dis = real_distance(ls[i[0]], ls[i[1]], max_area)  
            dis_2 = compute_distance(ls[i[0]], ls[i[1]])
            threshold = abs(ls[i[0]][0] - ls[i[0]][2]) + abs(ls[i[1]][0] - ls[i[1]][2])
            threshold_1 = abs(ls[i[0]][1] - ls[i[0]][3]) + abs(ls[i[1]][1] - ls[i[1]][3])
            rate_1 = compute_area(ls[i[0]])/max_area
            rate_2 = compute_area(ls[i[1]])/max_area
            # print(i, int(dis), int(threshold), threshold_1, rate_1, rate_2)
            # print("----------------")
            if dis < threshold*1.6 and dis_2 > 23 and (rate_1 > 0.37 and rate_2 > 0.37):

                df.loc[df['fname'] == name.replace("txt","jpg"), ['Distancing']] = 0
                count += 1
                break
        
    print(count)
    df.to_csv(r"C:\NTT\Zalo_5K\result\submission.csv", index=False)

label = r"C:\Users\trung\Downloads\zalo_AI\exp1\labels"
compute_distancing(label)