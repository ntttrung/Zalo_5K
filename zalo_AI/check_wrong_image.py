from ntpath import join
import cv2
import glob
import pandas as pd
import os
import numpy as np  
import math
import itertools  

APS = 1300 * 1600
color = (0, 255, 0)
thickness = 2

source = r'C:\Users\trung\Downloads\AI_zalo_5K\public_test\images'
label = r"C:\Users\trung\Downloads\zalo_AI\exp1\labels"

list_path = r"C:\Users\trung\Downloads\zalo_AI\exp1\labels\*.txt"
list_path = glob.glob(list_path)

lis_img = open(r"C:\Users\trung\Downloads\zalo_AI\list_img.txt","r")



def compute_distance(cor1, cor2):
    return math.sqrt((cor1[0] - cor2[0])**2 + (cor1[1] - cor2[1])**2)


def compute_angle(x1, x2):
    delta_x = abs(x1[0] - x2[0])
    delta_y = abs(x1[1] - x2[1])
    try:
        angle = math.atan(delta_y/delta_x)
    except:
        angle = 0
    return angle

def ratio_area(ls1, ls2, max_area):
    s1 = compute_area(ls1)
    s2 = compute_area(ls2)
    if s1 > s2:
        return s1/s2
    return s2/s1

def compute_area(ls):
    # new_ls = ls
    # box = BBox2D(ls)
    return math.sqrt((ls[1]-ls[3])**2 * (ls[2] -ls[0])**2)


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
    return ratio**2.2 * (compute_distance(x1, x2)) 

# for i in lis_img:
#     print(i.strip())
for path in lis_img:
    name = os.path.basename(path)
    name = path.strip()
    # ind  = list_path.index(path)
    path_image = join(source, name)
    img = cv2.imread(path_image)
    # img = cv2.resize(img, (800,800))
    dh, dw, _ = img.shape
    ra = math.sqrt(dh*dw/APS)
    fl = open(join(label, name.replace("jpg","txt")), 'r')
    data = fl.readlines()
    fl.close()

    #convert corr
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
    ##normailize area
    max_area = max(list_area)

    for i in range(len(ls)):
        cv2.rectangle(img, (ls[i][0], ls[i][1]), (ls[i][2], ls[i][3]), (0, 0, 255), 1)
    
    permute = list(range(0, len(ls)))
    isdistance = 1
    for i in itertools.permutations(permute, 2):
        dis = real_distance(ls[i[0]], ls[i[1]], max_area)  
        dis_2 = compute_distance(ls[i[0]], ls[i[1]])
        threshold = abs(ls[i[0]][0] - ls[i[0]][2]) + abs(ls[i[1]][0] - ls[i[1]][2])
        threshold_1 = abs(ls[i[0]][1] - ls[i[0]][3]) + abs(ls[i[1]][1] - ls[i[1]][3])
        rate_1 = compute_area(ls[i[0]])/max_area
        rate_2 = compute_area(ls[i[1]])/max_area
        if dis < threshold*1.6 and dis_2 > 23 and (rate_1 > 0.37 and rate_2 > 0.37):
            print(i, int(dis), dis_2, rate_1, rate_2)
            start = tuple(calculateCentroid(ls[i[0]]))
            end = tuple(calculateCentroid(ls[i[1]]))
            cv2.line(img, start, end, color, thickness)

            break
    img = cv2.resize(img, (800,800))
    cv2.imshow("Image",img)
    print(name)
    key = cv2.waitKey(0)
    if key == 48:
        break
