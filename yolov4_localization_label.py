from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from car_loc import car_localization
from car_rec import car_recognition
import argparse

# 把detection的座標換成opencv畫框的格式
def convertBack(x, y, w, h, img):
    H, W = img.shape[:2]
    xmin = int(round(x - (w / 2))) if x > w/2 else 0
    xmax = int(round(x + (w / 2))) if (x+w/2) < (W-1)  else W-1
    ymin = int(round(y - (h / 2))) if y > h/2 else 0
    ymax = int(round(y + (h / 2))) if (y+h/2) < (H-1)  else H-1
    return xmin, ymin, xmax, ymax

# 把detection的座標值轉成另一張圖的座標值
def convertNewImage(detections, originH, originW, newH, newW):
    new_detections = []
    for detection in detections:
        new = list(detection)
        x = (float(detection[2][0]) / originW) * newW
        y = (float(detection[2][1]) / originH) * newH
        w = (float(detection[2][2]) / originW) * newW
        h = (float(detection[2][3]) / originH) * newH
        new[2] = (x, y, w, h)
        new_detections.append(new)

    return new_detections

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

# 把百分比轉換成座標
def convertDetection(x, y, w, h, newH, newW):
    x = float(x) * newW
    y = float(y) * newH
    w = float(w) * newW
    h = float(h) * newH

    return x, y, w, h

def buildNamesList(path):
    file = open(path, 'r')
    names = []
    for line in file.readlines():
        names.append(line.split('\n')[0])
    file.close()
    return names

# 把box截圖並儲存
def cvSaveBox(detection, img, save_path, video_name, frame_num, plate_number):
    H, W = img.shape[:2]
    # 取得百分比
    x, y, w, h = detection[2][0],\
        detection[2][1],\
        detection[2][2],\
        detection[2][3]
    # 將百分比換成像素座標
    x, y, w, h = convertDetection(x, y, w, h, H, W)
    # 將像素座標換成opencv格式
    xmin, ymin, xmax, ymax = convertBack(
        float(x), float(y), float(w), float(h), img)
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    # 將圖片截下來
    crop_img = img[ymin:ymax, xmin:xmax]
    #cv2.imshow('test', crop_img)
    cv2.imwrite('{}/{}_f{}_{}.jpg'.format(save_path, video_name, frame_num, plate_number), crop_img)


def detect_single_video():
    path = opt.source
    ext = opt.ext
    video_name = path.split('/')[-1][:-4]
    video_ext = path[-3::]

    loc = car_localization()

    # fps_start = time.time()
    cap = cv2.VideoCapture(path)
    width = int(cap.get(3))
    height = int(cap.get(4))

    names = buildNamesList(opt.names)

    frame_num = 0
    print("Starting the YOLO loop...")
    while cap.grab():
        if frame_num % opt.save_per_frame == 0:
            # prev_time = time.time()
            ret, frame_read = cap.retrieve()
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,(416,416),interpolation=cv2.INTER_LINEAR)
            loc_detection = loc.detect(frame_resized)
            loc_detection = convertPercent(loc_detection, 416, 416)
            # print(loc_detection)
            if not loc_detection:
                frame_num += 1
                continue
            labelTxt = open('{}/{}_f{}.txt'.format(opt.save_local, video_name, frame_num), 'w')
            cv2.imwrite('{}/{}_f{}.jpg'.format(opt.save_local, video_name, frame_num), frame_read)

            # 將每個偵測到的框寫入label
            for i in loc_detection:
                classNum = names.index(i[0])
                WriteString = "{} {} {} {} {}\n".format(classNum, i[2][0], i[2][1], i[2][2], i[2][3])
                # print(type(WriteString))
                labelTxt.write(WriteString)
                plate_number = 0
                # 如果是plate，則將plate擷取下來
                if i[0] == 'plate':
                    cvSaveBox(i, frame_read, opt.save_recog, video_name, frame_num, plate_number)
                    plate_number += 1
            labelTxt.close()

            # print FPS
            #print(1/(time.time()-prev_time))

            if opt.view_img:
                cv2.imshow('Demo', frame_read)
                cv2.waitKey(1)
        frame_num += 1
    cap.release()

def detect_multiple_video():
    # get path
    path = opt.source
    ext = opt.ext

    # load yolo
    loc = car_localization()

    names = buildNamesList(opt.names)

    for i in os.listdir(path):
        if i[-3::] == ext:
            # set video capture and video writer
            cap = cv2.VideoCapture(path+'/'+i)
            width = int(cap.get(3))
            height = int(cap.get(4))
            
            frame_num = 0
            video_name = i[0:-4]
            print("Starting video {}...".format(video_name))
            # detect video frame
            while cap.grab():
                if frame_num % opt.save_per_frame == 0:
                    # prev_time = time.time()
                    ret, frame_read = cap.retrieve()
                    frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb,(416,416),interpolation=cv2.INTER_LINEAR)
                    loc_detection = loc.detect(frame_resized)
                    loc_detection = convertPercent(loc_detection, 416, 416)
                    # print(loc_detection)
                    if not loc_detection:
                        frame_num += 1
                        continue
                    labelTxt = open('{}/{}_f{}.txt'.format(opt.save_local, video_name, frame_num), 'w')
                    cv2.imwrite('{}/{}_f{}.jpg'.format(opt.save_local, video_name, frame_num), frame_read)

                    # 將每個偵測到的框寫入label
                    for i in loc_detection:
                        classNum = names.index(i[0])
                        WriteString = "{} {} {} {} {}\n".format(classNum, i[2][0], i[2][1], i[2][2], i[2][3])
                        # print(type(WriteString))
                        labelTxt.write(WriteString)
                        plate_number = 0
                        # 如果是plate，則將plate擷取下來
                        if i[0] == 'plate':
                            cvSaveBox(i, frame_read, opt.save_recog, video_name, frame_num, plate_number)
                            plate_number += 1
                    labelTxt.close()

                    # print FPS
                    # print("FPS:", 1/(time.time()-prev_time))
                    if opt.view_img:
                        cv2.imshow('Demo', frame_read)
                        cv2.waitKey(1)
                frame_num += 1
            cap.release()
            print("Finish video {}.".format(video_name))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='', help='source')
    parser.add_argument('--names', type=str, default='', help='localization names file path')
    parser.add_argument('--ext', type=str, default='mp4', help='extension of save data')
    parser.add_argument('--save-per-frame', type=int, default=10, help='how many frames save one label')
    parser.add_argument('--save-local', type=str, default='./localization', help='output folder')
    parser.add_argument('--save-recog', type=str, default='./recognition', help='output folder')
    parser.add_argument('--view-img', action='store_true', help='display results')

    opt = parser.parse_args()
    print(opt)

    start_time = time.time()

    # 如果沒有輸入names位置則結束程式
    opt.names = opt.names.replace('\\', '/')
    if not os.path.isfile(opt.names):
        print("\nError. Your path of names is not a file.")
        sys.exit()

    # 建立output資料夾
    opt.save_local = opt.save_local.replace('\\', '/')
    if not os.path.isdir(opt.save_local):
        os.mkdir(opt.save_local)
        print("\nMake {} directory done!".format(opt.save_local))
    opt.save_recog = opt.save_recog.replace('\\', '/')
    if not os.path.isdir(opt.save_recog):
        os.mkdir(opt.save_recog)
        print("\nMake {} directory done!".format(opt.save_recog))

    # 檢查是否為single
    opt.source = opt.source.replace('\\', '/')
    if os.path.isdir(opt.source):
        detect_multiple_video()
    elif os.path.isfile(opt.source):
        detect_single_video()

    end_time = time.time()
    print("\nFinish! Cost time is {}s.".format(end_time - start_time))