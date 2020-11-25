from ctypes import *
import os
import numpy as np
import time
import darknet

class car_localization(object):
    def __init__(self):
        self.config_file = "/home/ronfrog/yolov4/darknet/car_demo/Localization_yolov4_cfg/yolov4-tiny.cfg"
        self.weights = "/home/ronfrog/yolov4/darknet/car_demo/Localization_yolov4_cfg/weights/yolov4-tiny_best.weights"
        self.data_file = "/home/ronfrog/yolov4/darknet/car_demo/Localization_yolov4_cfg/localization.data"

        self.network, self.class_names, self.class_colors = darknet.load_network(
            self.config_file,
            self.data_file,
            self.weights,
            batch_size=1
        )

    def detect(self,img):
        width = darknet.network_width(self.network)
        height = darknet.network_height(self.network)
        darknet_image = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(darknet_image,img.tobytes())
        detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=0.7)
        darknet.free_image(darknet_image)
        return detections

