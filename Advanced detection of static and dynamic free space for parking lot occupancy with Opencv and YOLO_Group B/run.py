import numpy as np
import argparse
import cv2
import subprocess


import lib
import yolo
import yolo_utils

VIDEO_SOURCE = cv2.VideoCapture(r'D:\Uni Doc\DCAITI Projekt\Dokumentation\Free-space-detection-4\RAW_data\josn_file_foto\c7753.jpg')

if __name__ == '__main__':
    vid,_ = yolo.run_yolo_(VIDEO_SOURCE)
    yolo_utils.show_image(vid)
