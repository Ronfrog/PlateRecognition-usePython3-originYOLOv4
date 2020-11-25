from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from car_rec import car_recognition
import argparse
from tqdm import tqdm

# 把detection的座標換成百分比
def convertPercent(detections, originH, originW):
    new_detections = []
    for detection in detections:
        new = list(detection)
        x = float(detection[2][0]) / originW
        y = float(detection[2][1]) / originH
        w = float(detection[2][2]) / originW
        h = float(detection[2][3]) / originH
        new[2] = (x, y, w, h)
        new_detections.append(new)

    return new_detections

def buildNamesList(path):
    file = open(path, 'r')
    names = []
    for line in file.readlines():
        names.append(line.split('\n')[0])
    file.close()
    return names

def detect_directory():
    # get path
    path = opt.source
    ext = opt.ext

    # load yolo
    rec = car_recognition()

    names = buildNamesList(opt.names)

    for i in tqdm(os.listdir(path)):
        if i[-3::] == ext:
            # set video capture and video writer
            img = cv2.imread(path+'/'+i)
            height , width = img.shape[:2]

            img_name = i[0:-4]
            # detect
            # prev_time = time.time()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb,(128,128),interpolation=cv2.INTER_LINEAR)
            _, rec_detection = rec.detect(img_resized)
            #print(rec_detection)
            rec_detection = convertPercent(rec_detection, 128, 128)
            
            if not rec_detection:
                continue
            labelTxt = open('{}/{}.txt'.format(path, img_name), 'w')

            # 將每個偵測到的框寫入label
            for i in rec_detection:
                classNum = names.index(i[0])
                WriteString = "{} {} {} {} {}\n".format(classNum, i[2][0], i[2][1], i[2][2], i[2][3])
                # print(type(WriteString))
                labelTxt.write(WriteString)
            labelTxt.close()

            # print FPS
            # print("FPS:", 1/(time.time()-prev_time))
            if opt.view_img:
                cv2.imshow('Demo', frame_read)
                cv2.waitKey(1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='', help='source')
    parser.add_argument('--names', type=str, default='', help='localization names file path')
    parser.add_argument('--ext', type=str, default='jpg', help='extension of save data')
    parser.add_argument('--view-img', action='store_true', help='display results')

    opt = parser.parse_args()
    print(opt)

    start_time = time.time()

    # 如果沒有輸入names位置則結束程式
    opt.names = opt.names.replace('\\', '/')
    if not os.path.isfile(opt.names):
        print("\nError. Your path of names is not a file.")
        sys.exit()

    detect_directory()

    end_time = time.time()
    print("\nFinish! Cost time is {}s.".format(end_time - start_time))