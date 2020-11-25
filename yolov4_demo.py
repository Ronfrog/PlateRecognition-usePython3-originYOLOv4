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
        x = (detection[2][0] / originW) * newW
        y = (detection[2][1] / originH) * newH
        w = (detection[2][2] / originW) * newW
        h = (detection[2][3] / originH) * newH
        new[2] = (x, y, w, h)
        new_detections.append(new)

    return new_detections

def drawplate_rec(detections, x, y, w, h, img):
    xmin = int(round(x - (w / 2)))
    ymax = int(round(y + (h / 2)))
    for i in detections:
        cv2.putText(img,str(i[0]),(xmin,ymax+5),cv2.FONT_HERSHEY_SIMPLEX, 0.5,[255, 0, 0], 2)
        xmin = xmin + 10
    return img

def cvDrawBoxes(detections, img):
    color_plate = [(0,255,0),(255,0,0)]
    for detection in detections:
        #print(detection)
        if detection[0] == 'plate':
            color = color_plate[1]
        else:
            color = color_plate[0]
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h), img)
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, color, 1)
        cv2.putText(img,
                    str(detection[0]) +
                    " [" + str(round(float(detection[1]), 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)
    return img

# 把box截圖並儲存
def cvSaveBox(detections, img, video_name, frame_num):
    color_plate = [(0,255,0),(255,0,0)]
    number = 0
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h), img)
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        
        crop_img = img[ymin:ymax, xmin:xmax]
        #cv2.imshow('test', crop_img)
        cv2.imwrite('./crop_pic/{}_f{}_{}.jpg'.format(video_name, frame_num, number), crop_img)
        number += 1
        

def crop_plate(x,y,w,h,img):
    xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h), img)
    crop_img = img[ymin:ymax,xmin:xmax]
    return crop_img

def detect_single_video():
    path = opt.source.replace('\\', '/')
    ext = opt.ext
    video_name = path.split('/')[-1][:-4]

    loc = car_localization()
    rec = car_recognition()

    output_path = opt.output.replace('\\', '/')
    output_path = output_path+'/'+video_name+'.'+'avi'
    fps_start = time.time()
    cap = cv2.VideoCapture(path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    # cap.set(3, 1280)
    # cap.set(4, 720)
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"MJPG"), 20.0,
        (width,height))
    print("Starting the YOLO loop...")
    while cap.grab():
        prev_time = time.time()
        ret, frame_read = cap.retrieve()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,(416,416),interpolation=cv2.INTER_LINEAR)
        loc_detection = loc.detect(frame_resized)
        loc_detection = convertNewImage(loc_detection, 416, 416, height, width)
        image = cvDrawBoxes(loc_detection, frame_rgb)
        # print(loc_detection)
        for i in loc_detection:
            if i[0] == 'plate':
                plate_img = crop_plate(i[2][0], i[2][1], i[2][2], i[2][3], frame_rgb)
                plate_img = cv2.resize(plate_img, (128,128), interpolation=cv2.INTER_LINEAR)
                rec_detection, predict= rec.detect(plate_img)
                image = drawplate_rec(rec_detection, i[2][0], i[2][1], i[2][2], i[2][3], image)

        # fps_end = time.time()
        # fps = 1/(fps_end - fps_start)
        #print(1/(time.time()-fps_start))

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)
        if opt.view_img:
            cv2.imshow('Demo', image)
            cv2.waitKey(1)
    cap.release()
    out.release()
    darknet.free_image(darknet_image)

def detect_file_video():
    # get path
    path = opt.source.replace('\\', '/')
    ext = opt.ext
    output_path = opt.output.replace('\\', '/')

    # load yolo
    #darknet_image = darknet.make_image(1280,720,3)
    #loc = car_localization(darknet_image)
    loc = car_localization()
    rec = car_recognition()

    for i in os.listdir(path):
        if i[-3::] == ext:
            # set video capture and video writer
            save_path = output_path+'/'+i[0:-4]+"."+'avi'
            cap = cv2.VideoCapture(path+'/'+i)
            width = int(cap.get(3))
            height = int(cap.get(4))
            out = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"MJPG"), 20.0,
                (width,height))
            frame_num = 0
            video_name = i[0:-4]
            print("Starting the YOLO loop...")
            # detect video frame
            while cap.grab():
                prev_time = time.time()
                ret, frame_read = cap.retrieve()
                frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb,(416,416),interpolation=cv2.INTER_LINEAR)
                loc_detection = loc.detect(frame_resized)
                loc_detection = convertNewImage(loc_detection, 416, 416, height, width)
                image = cvDrawBoxes(loc_detection, frame_resized)
                # cvSaveBox(loc_detection, frame_resized, video_name, frame_num)

                for i in loc_detection:
                    if i[0] == 'plate':
                        plate_img = crop_plate(i[2][0],i[2][1],i[2][2],i[2][3],frame_resized)
                        plate_img = cv2.resize(plate_img,(128,128),interpolation=cv2.INTER_LINEAR)
                        # cv2.imshow('crop', plate_img)
                        rec_detection ,predict= rec.detect(plate_img)
                        image = drawplate_rec(rec_detection,i[2][0],i[2][1],i[2][2],i[2][3],image)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # print FPS
                print("FPS:", 1/(time.time()-prev_time))
                out.write(image)
                if opt.view_img:
                    cv2.imshow('Demo', image)
                    cv2.waitKey(1)
                frame_num += 1
            cap.release()
            out.release()
    darknet.free_image(darknet_image)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='', help='source')
    parser.add_argument('--output', type=str, default='./output', help='output folder')
    parser.add_argument('--ext', type=str, default='mp4', help='extension of save data')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--single', action='store_true', help='single video or videos in a file')
    opt = parser.parse_args()
    print(opt)

    if opt.single:
        if os.path.isdir(opt.source):
            print("Error. Source path is a directory. Don't use 'single' argument.")
            sys.exit()
        detect_single_video()
    else:
        if os.path.isfile(opt.source):
            print("Error. Source path is a file. Should use 'single' argument.")
            sys.exit()
        detect_file_video()
