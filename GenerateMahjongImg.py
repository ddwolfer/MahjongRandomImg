import os
import cv2
import numpy as np
import shutil
import random
import argparse
import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--images_num", type=int, default=100)
parser.add_argument("--image_size", type=int, default=416)
parser.add_argument("--source_path", type=str, default="./images")
parser.add_argument("--output_path", type=str, default="./output")
parser.add_argument("--labels_txt", type=str, default="./labels.txt")
parser.add_argument("--small", type=int, default=3)
parser.add_argument("--medium", type=int, default=6)
parser.add_argument("--big", type=int, default=3)
flags = parser.parse_args()


#初始設置
images_num = flags.images_num
SIZE = flags.image_size
images_output_path = flags.output_path
image_source_path = flags.source_path
labels_txt_name = flags.labels_txt
small_num = flags.small
medium_num = flags.medium
big_num = flags.big

# 把輸出的資料夾淨空
if os.path.exists(images_output_path): 
    shutil.rmtree(images_output_path)
os.mkdir(images_output_path)

#確認圖片生成有沒有衝突
def compute_iou(box1, box2):
    """xmin, ymin, xmax, ymax"""

    A1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    A2 = (box2[2] - box2[0])*(box2[3] - box2[1])

    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    if ymin >= ymax or xmin >= xmax: return 0
    return  ((xmax-xmin) * (ymax - ymin)) / (A1 + A2)

#回傳麻將的ID
def get_mahjong_ID(jpgName):
    getCSV = pd.read_csv('data.csv',index_col="image-name")
    return getCSV.loc[jpgName].label

#做圖片放上去背景
def make_image(data, image_path, ratio=1):
    #抓目前的狀況
    blank = data[0]
    boxes = data[1]
    label = data[2]
    #看是哪個ID
    ID = get_mahjong_ID(image_path.split("/")[-1])
    #抓圖片
    image = cv2.imread(image_path)
    #放大/縮小
    image = cv2.resize(image, (int(24*ratio), int(32*ratio)))
    #拿圖片的hright width
    h, w, c = image.shape
    #如果跟其他圖片有衝突 換個位置
    while True:
        xmin = np.random.randint(0, SIZE-w, 1)[0]
        ymin = np.random.randint(0, SIZE-h, 1)[0]
        xmax = xmin + w
        ymax = ymin + h
        box = [xmin, ymin, xmax, ymax]

        iou = [compute_iou(box, b) for b in boxes]
        if max(iou) == 0.:
            boxes.append(box)
            label.append(ID)
            break
    #把圖片放上去
    for i in range(w):
        for j in range(h):
            x = xmin + i
            y = ymin + j
            blank[y][x] = image[j][i]
    # 如果想要框框可以放這個 cv2.rectangle(blank, (xmin, ymin), (xmax, ymax), [0, 0, 255], 2)
    # 把這個副函式弄好的傳回去
    data[0] = blank
    data[1] = boxes
    data[2] = label
    return data

image_paths  = [ "./images/" + image_name for image_name in os.listdir(image_source_path)]

with open(labels_txt_name, "w") as wf:
    #算現在跑幾張了
    image_count = 0
    while image_count < images_num: #跑上面設置的總數
        image_path = os.path.realpath(os.path.join(images_output_path, "%04d.jpg" %(image_count+1))) #輸出名字
        annotation = image_path 
        blanks = np.ones(shape=[SIZE, SIZE, 3]) * 255 #白色背景大小
        bboxes = [[0,0,1,1]] #XY位置
        labels = [0] #標籤
        data = [blanks, bboxes, labels]
        
        #圖片START
        # small object 生小
        ratios = [0.6, 0.8]
        N = random.randint(0, small_num)
        for _ in range(N):
            ratio = random.choice(ratios)
            idx = random.randint(1, 627)
            data = make_image(data, image_paths[idx], ratio)
            
        # medium object 生中
        ratios = [1., 1.25, 1.5, 1.75]
        N = random.randint(0, big_num)
        for _ in range(N):
            ratio = random.choice(ratios)
            idx = random.randint(1, 627)
            data = make_image(data, image_paths[idx], ratio)
            
        # medium object 生大
        ratios = [2. , 2.5, 3.]
        N = random.randint(0, medium_num)
        for _ in range(N):
            ratio = random.choice(ratios)
            idx = random.randint(1, 627)
            data = make_image(data, image_paths[idx], ratio)
        #圖片 END
        #生output圖
        cv2.imwrite(image_path, data[0])
        #生txt 檔案
        for i in range(len(labels)):
            if i == 0: continue #第0張是背景
            xmin = str(bboxes[i][0])
            ymin = str(bboxes[i][1])
            xmax = str(bboxes[i][2])
            ymax = str(bboxes[i][3])
            class_ind = str(labels[i])
            annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            
        
        image_count += 1
        print("=> %s" %annotation)
        wf.write(annotation + "\n")