import pandas as pd
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

df = pd.read_csv(r".\model_pretrain_coco.csv")
# for path in list_path:
#     name = os.path.basename(path)
#     ind  = list_path.index(path)
    path_image = join(source, name.replace("txt","jpg"))
#     print(name.replace("txt","jpg"))
print(df[df['fname'] == '11.jpg'])
