import cv2 as cv
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import sys
import argparse
import math
from datetime import datetime

## All loaded images should be with the size: 640/360 pixels!!

## Load an color image in grayscale.
image_path_read = 'Input/' + 'CAM_1_center.jpeg' # day # relative path to the folder "Input"
#image_path_read = 'Input/' + 'CAM_1c_1588271652_resized.jpg' # day-night
#image_path_read = 'Input/' + 'CAM_1c_1588273684_resized.jpg' # night
#image_path_read = 'Input/' + 'CAM_1c_1589894225_resized.jpg' # day, a lot of parked vehicles
#image_path_read = 'Input/' + 'CAM_1c_1589887584_resized.jpg' # day, a lot of vehicles, moving car in front of a parking slot 1
#image_path_read = 'Input/' + 'CAM_1c_1589887389_resized.jpg' # day, a lot of vehicles, moving car in front of a parking slot 2
#image_path_read = 'Input/' + 'CAM_1c_1589887409_resized.jpg' # day, a lot of vehicles, moving car in front of a parking slot 3.1
#image_path_read = 'Input/' + 'CAM_1c_1589887410_resized.jpg' # day, a lot of vehicles, moving car in front of a parking slot 3.2
#image_path_read = 'Input/' + 'CAM_1c_1589887412_resized.jpg' # day, a lot of vehicles, moving car in front of a parking slot 3.3
#image_path_read = 'Input/' + 'CAM_1c_1589888659_resized.jpg' # day, a lot of vehicles, moving car in front of a parking slot 4.1
#image_path_read = 'Input/' + 'CAM_1c_1589888660_resized.jpg' # day, a lot of vehicles, moving car in front of a parking slot 4.2
# traffic jam -> take more then 3 pictures one after another, or have a plan B and consider only the center of the parking space
#image_path_read = 'Input/' + 'CAM_1c_1591887712_resized.jpg' # bad weather, traffic jam, a car in front 1
#image_path_read = 'Input/' + 'CAM_1c_1591887774_resized.jpg' # bad weather, traffic jam, a car in front 2
#image_path_read = 'Input/' + 'CAM_1c_1591887881_resized.jpg' # even worse weather, traffic jam, a car in front 3
#image_path_read = 'Input/' + 'CAM_1c_1591851682_resized.jpg' # bad weather 1.1 (pictures one after another)
#image_path_read = 'Input/' + 'CAM_1c_1591851683_resized.jpg' # bad weather 1.2
image_original = cv.imread(image_path_read)
height, width = image_original.shape[:2]
h_scenario2_top = 84
h_scenario2_bottom = 110

w_scenario3_first_part_left = 1
w_scenario3_first_part_right = 55
w_scenario3_second_part_left = 189 # the same start as the static area
w_scenario3_second_part_right = width - 2
h_scenario3_top = 65
h_scenario3_bottom = 75

## load 2-3 pictures one after another, make the tests on all of them and after that make a decission (for area 1):
image1 = 'CAM_1c_1589888659_resized.jpg';
image2 = 'CAM_1c_1589888660_resized.jpg';

def set_img_CAM_center(one_picture):
    if one_picture == True: # test only 1 image
        image_path_read_array = np.array([image_path_read])
    else: # test a sequence of images (2-3 one after another)
        image_path_read_array = np.array([image1, image2])
    return image_path_read_array

def img_CAM_center(one_picture, count):
    image_path_read_array = set_img_CAM_center(one_picture)
    return cv.imread(image_path_read_array[count-1])

def color_detection(img, lower, upper, color_change):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) # convert to hsv
    mask = cv.inRange(img, lower, upper)
    if color_change != -1:
        img[mask>0] = color_change # change the color of a group of pixels
    img_color_detection = cv.bitwise_and(img, img, mask = mask)
    mask_invert = cv.bitwise_not(mask)
    return img_color_detection, mask_invert

def draw_contours(img, contours_color):
    imggray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imggray,127,255,0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) # RETR_TREE/RETR_EXTERNAL # CHAIN_APPROX_SIMPLE/CHAIN_APPROX_NONE
    img_with_contours = cv.drawContours(img, contours, -1, contours_color, 1)
    return img_with_contours

def mask_rectangle_scenario2(img):
    black_background = np.zeros((image_original.shape[0], image_original.shape[1], 3), np.uint8) #---black in RGB
    black1_background = cv.rectangle(black_background,(1, h_scenario2_top),(width-2, h_scenario2_bottom),(255, 255, 255), -1) #---the dimension of the ROI (region of interest)
    gray_background = cv.cvtColor(black_background,cv.COLOR_BGR2GRAY) #---converting to gray
    img_mask_rectangle = cv.bitwise_and(img, img, mask=gray_background)
    return img_mask_rectangle

def mask_polygon_scenario2(img, scenario2_polygon_corners):
    black_background = np.zeros((image_original.shape[0], image_original.shape[1], 3), np.uint8) #---black in RGB
    black_background1 = cv.fillPoly(black_background, [scenario2_polygon_corners], (255, 255, 255)) #---the dimension of the ROI (region of interest)
    gray_background = cv.cvtColor(black_background,cv.COLOR_BGR2GRAY) #---converting to gray
    img_mask_polygon = cv.bitwise_and(img, img, mask=gray_background)
    return img_mask_polygon

def mask_polygon_inv_scenario2(img, scenario2_polygon_corners):
    black_background = np.zeros((image_original.shape[0], image_original.shape[1], 3), np.uint8) #---black in RGB
    black_background1 = cv.fillPoly(black_background, [scenario2_polygon_corners], (255, 255, 255)) #---the dimension of the ROI (region of interest)
    gray_background = cv.cvtColor(black_background,cv.COLOR_BGR2GRAY) #---converting to gray
    gray_background_inv = cv.bitwise_not(gray_background)
    img_mask_polygon_inv = cv.bitwise_and(img, img, mask=gray_background_inv)
    return img_mask_polygon_inv

def mask_rectangle_scenario3(img):
    black_background = np.zeros((image_original.shape[0], image_original.shape[1], 3), np.uint8) #---black in RGB
    black1_background = cv.rectangle(black_background,(w_scenario3_first_part_left, h_scenario3_top),(w_scenario3_first_part_right, h_scenario3_bottom),(255, 255, 255), -1) #---the dimension of the ROI (region of interest)
    gray_background = cv.cvtColor(black_background,cv.COLOR_BGR2GRAY) #---converting to gray
    img_mask_rectangle_1 = cv.bitwise_and(img, img, mask=gray_background)
    
    black_background = np.zeros((image_original.shape[0], image_original.shape[1], 3), np.uint8) #---black in RGB
    black2_background = cv.rectangle(black_background,(w_scenario3_second_part_left, h_scenario3_top),(w_scenario3_second_part_right, h_scenario3_bottom),(255, 255, 255), -1) #---the dimension of the ROI (region of interest)
    gray2_background = cv.cvtColor(black_background,cv.COLOR_BGR2GRAY) #---converting to gray
    img_mask_rectangle_2 = cv.bitwise_and(img, img, mask=gray2_background)

    img_mask_rectangle = cv.bitwise_or(img_mask_rectangle_1, img_mask_rectangle_2)
    
    return img_mask_rectangle

def mask_rectangle_inv_scenario3(img):
    black_background = np.zeros((image_original.shape[0], image_original.shape[1], 3), np.uint8) #---black in RGB
    black1_background = cv.rectangle(black_background,(w_scenario3_first_part_left, h_scenario3_top),(w_scenario3_first_part_right, h_scenario3_bottom),(255, 255, 255), -1) #---the dimension of the ROI (region of interest)
    gray_background = cv.cvtColor(black_background,cv.COLOR_BGR2GRAY) #---converting to gray
    
    black_background = np.zeros((image_original.shape[0], image_original.shape[1], 3), np.uint8) #---black in RGB
    black2_background = cv.rectangle(black_background,(w_scenario3_second_part_left, h_scenario3_top),(w_scenario3_second_part_right, h_scenario3_bottom),(255, 255, 255), -1) #---the dimension of the ROI (region of interest)
    gray1_background = cv.cvtColor(black_background,cv.COLOR_BGR2GRAY) #---converting to gray

    gray2_background = cv.bitwise_or(gray_background, gray1_background)
    gray2_background_inv = cv.bitwise_not(gray2_background)
    img_mask_rectangle_inv = cv.bitwise_and(img, img, mask=gray2_background_inv)
    return img_mask_rectangle_inv

def marked_static_parking_slots(img):
    ## mark approximately the contours of the static parking area (rectangle):
    h_area2_top = 84
    h_area2_bottom = 110
    ## mark exactly the contours of the static parking area (polygon):
    h_bottom_left = 104
    h_top_left = 87
    h_top_right = 89
    h_bottom_right = 108
    ## find the angle of the static parking spaces -> tune the lines:
    height_slot = h_area2_bottom - h_area2_top
    width_slot = 19
    width_shift = 13
    width_start = 1
    w_second_part = 189
    width_start = w_second_part
    width_end = width - 2
    c = 0
    width_slot_2 = width_slot + 1
    width_slot_3 = width_slot - 1
    while width_start < width_end:
        if c == 0:
            width_temp_1 = width_start+width_shift
            h_temp_1 = h_bottom_left
            width_temp_2 = width_start
            h_temp_2 = h_top_left
        elif 1 <= c < 3:
            width_start += width_slot
            width_temp_1 = width_start+width_shift
            h_temp_1 = h_bottom_left
            width_temp_2 = width_start
            h_temp_2 = h_top_left
        elif 3 <= c < 6:
            width_start += width_slot_2
            width_temp_1 = width_start+width_shift+5
            h_temp_1 = h_bottom_left+1
            width_temp_2 = width_start
            h_temp_2 = h_top_left
        elif 6 <= c < 8:
            width_start += width_slot_2
            width_temp_1 = width_start+width_shift+6
            h_temp_1 = h_bottom_left+1
            width_temp_2 = width_start
            h_temp_2 = h_top_left+1
        elif 8 <= c < 14:
            width_start += width_slot_2
            width_temp_1 = width_start+width_shift+10
            h_temp_1 = h_bottom_left+2
            width_temp_2 = width_start
            h_temp_2 = h_top_left+1
        elif 14 <= c < 18:
            width_start += width_slot
            width_temp_1 = width_start+width_shift+12
            h_temp_1 = h_bottom_left+3
            width_temp_2 = width_start
            h_temp_2 = h_top_left+1
        elif 18 <= c < 19:
            width_start += width_slot_2
            width_temp_1 = width_start+width_shift+12
            h_temp_1 = h_bottom_left+3
            width_temp_2 = width_start
            h_temp_2 = h_top_left+2
        elif 19 <= c < 20:
            width_start += width_slot_3
            width_temp_1 = width_start+width_shift+16
            h_temp_1 = h_bottom_left+4
            width_temp_2 = width_start
            h_temp_2 = h_top_left+2
        elif 20 <= c < 22:
            width_start += width_slot_3
            width_temp_1 = width_start+width_shift+17
            h_temp_1 = h_bottom_left+4
            width_temp_2 = width_start
            h_temp_2 = h_top_left+2
        else:
            width_start += width_slot
            width_temp_1 = 0
            h_temp_1 = 0
            width_temp_2 = 0
            h_temp_2 = 0
        cv.line(img, (width_temp_1, h_temp_1), (width_temp_2, h_temp_2), (255, 0, 0), thickness=1)
        c += 1
    return img

def color_boundaries(pixels_test_color,one_picture,counter_images):
    colorB_min = 255
    colorB_max = 0
    colorG_min = 255
    colorG_max = 0
    colorR_min = 255
    colorR_max = 0
    for pixel in pixels_test_color:
        if int(img_CAM_center(one_picture,counter_images)[pixel[0], pixel[1], 0]) > colorB_max:
            colorB_max = int(img_CAM_center(one_picture,counter_images)[pixel[0], pixel[1], 0])
        if int(img_CAM_center(one_picture,counter_images)[pixel[0], pixel[1], 0]) < colorB_min:
            colorB_min = int(img_CAM_center(one_picture,counter_images)[pixel[0], pixel[1], 0])

        if int(img_CAM_center(one_picture,counter_images)[pixel[0], pixel[1], 1]) > colorG_max:
            colorG_max = int(img_CAM_center(one_picture,counter_images)[pixel[0], pixel[1], 1])
        if int(img_CAM_center(one_picture,counter_images)[pixel[0], pixel[1], 1]) < colorG_min:
            colorG_min = int(img_CAM_center(one_picture,counter_images)[pixel[0], pixel[1], 1])

        if int(img_CAM_center(one_picture,counter_images)[pixel[0], pixel[1], 2]) > colorR_max:
            colorR_max = int(img_CAM_center(one_picture,counter_images)[pixel[0], pixel[1], 2])
        if int(img_CAM_center(one_picture,counter_images)[pixel[0], pixel[1], 2]) < colorR_min:
            colorR_min = int(img_CAM_center(one_picture,counter_images)[pixel[0], pixel[1], 2])

    #print(colorB_min) # 106
    #print(colorB_max) # 141
    #print(colorG_min) # 97
    #print(colorG_max) # 126
    #print(colorR_min) # 88
    #print(colorR_max) # 141

    # additionally round the values:
    if colorB_min < colorG_min:
        colorG_min = colorB_min
    else:
        colorB_min = colorG_min
    
    if colorR_min < colorG_min:
        colorG_min = colorR_min
    else:
        colorR_min = colorG_min
    
    if colorB_min < colorR_min:
        colorR_min = colorB_min
    else:
        colorB_min = colorR_min
    
    if colorB_max > colorG_max:
        colorG_max = colorB_max
    else:
        colorB_max = colorG_max
    
    if colorR_max > colorG_max:
        colorG_max = colorR_max
    else:
        colorR_max = colorG_max
    
    if colorB_max > colorR_max:
        colorR_max = colorB_max
    else:
        colorB_max = colorR_max

    #print(colorB_min) # 88
    #print(colorB_max) # 141

    # additionally round the values (2):
    colorB_min = colorB_min - 5
    colorG_min = colorB_min - 5
    colorR_min = colorB_min - 5
    colorB_max = colorB_max + 5
    colorG_max = colorG_max + 5
    colorR_max = colorR_max + 5

    lower = np.array([colorB_min, colorG_min, colorR_min], dtype = "uint8") # lower boundary # shadows!
    upper = np.array([colorB_max, colorG_max, colorR_max], dtype = "uint8") # upper boundary # shadows!

    return lower, upper

def color_test_points(image_temp):
    image_temp[43, 283] = (0,0,255) # find if it is a day or night
    #/*******************************************************/
    image_temp[90, 36] = (255,0,255) # static parking boarders
    image_temp[91, 34] = (255,0,255)
    image_temp[92, 31] = (255,0,255)

    image_temp[90, 210] = (0,0,0) # shadows
    image_temp[90, 280] = (0,0,0)
    image_temp[90, 300] = (0,0,0)
    image_temp[100, 203] = (0,0,0)
    image_temp[103, 208] = (0,0,0)
    image_temp[91, 238] = (0,0,0)
    image_temp[91, 283] = (0,0,0)
    image_temp[95, 215] = (0,0,0)
    image_temp[95, 216] = (0,0,0)
    image_temp[95, 275] = (0,0,0)
    image_temp[95, 280] = (0,0,0)
    image_temp[95, 286] = (0,0,0)
    image_temp[92, 390] = (0,0,0)
    image_temp[93, 256] = (0,0,0)
    image_temp[93, 255] = (0,0,0)
    image_temp[90, 200] = (0,255,255) # vehicle instead of a shadow
    image_temp[90, 203] = (0,255,255)
    image_temp[91, 205] = (0,255,255)
    image_temp[93, 207] = (0,255,255)
    image_temp[91, 400] = (0,255,255)
    image_temp[93, 407] = (0,255,255)

    image_temp[100, 410] = (0,0,0) # inside of a parking space
    image_temp[95, 400] = (0,0,0)
    image_temp[100, 430] = (0,0,0)
    image_temp[95, 430] = (0,0,0)
    image_temp[103, 250] = (0,0,0)
    image_temp[100, 242] = (0,0,0)
    image_temp[100, 515] = (0,0,0)
    image_temp[105, 515] = (0,0,0)
    image_temp[95, 590] = (0,0,0)
    image_temp[105, 590] = (0,0,0)
    image_temp[88, 273] = (0,0,0)
    image_temp[100, 275] = (0,0,0)
    image_temp[100, 305] = (0,0,0)
    image_temp[90, 305] = (0,0,0)
    #/*******************************************************/
    image_temp[80, 290] = (0,0,255) # lamps
    image_temp[80, 295] = (0,0,255)
    image_temp[80, 300] = (0,0,255)
    image_temp[80, 305] = (0,0,255)
    image_temp[80, 310] = (0,0,255)
    image_temp[80, 315] = (0,0,255)
    image_temp[80, 320] = (0,0,255)
    image_temp[77, 290] = (0,0,255)
    image_temp[77, 295] = (0,0,255)
    image_temp[77, 300] = (0,0,255)
    image_temp[77, 305] = (0,0,255)
    image_temp[77, 310] = (0,0,255)
    image_temp[77, 315] = (0,0,255)
    image_temp[77, 320] = (0,0,255)

    image_temp[78, 405] = (0,0,255) # left/right from the next lamp
    image_temp[78, 410] = (0,0,255)
    image_temp[78, 415] = (0,0,255)
    image_temp[78, 420] = (0,0,255)
    image_temp[78, 425] = (0,0,255)
    image_temp[78, 430] = (0,0,255)
    image_temp[79, 460] = (0,0,255)
    image_temp[79, 465] = (0,0,255)
    image_temp[79, 470] = (0,0,255)
    image_temp[79, 475] = (0,0,255)
    image_temp[79, 480] = (0,0,255)
    image_temp[79, 485] = (0,0,255)
    #/*******************************************************/
    image_temp[77, 350] = (0,0,255) # shadows 2
    image_temp[77, 355] = (0,0,255)
    image_temp[77, 360] = (0,0,255)
    image_temp[77, 365] = (0,0,255)
    image_temp[77, 370] = (0,0,255)
    image_temp[77, 375] = (0,0,255)
    image_temp[77, 380] = (0,0,255)
    #/*******************************************************/
    image_temp[55, 220] = (255,0,0) # trees
    image_temp[55, 230] = (255,0,0)
    image_temp[55, 240] = (255,0,0)
    image_temp[55, 250] = (255,0,0)
    image_temp[45, 220] = (255,0,0)
    image_temp[45, 230] = (255,0,0)
    image_temp[45, 240] = (255,0,0)
    image_temp[45, 250] = (255,0,0)
    image_temp[45, 260] = (255,0,0)
    image_temp[35, 220] = (255,0,0)
    image_temp[35, 230] = (255,0,0)
    image_temp[35, 240] = (255,0,0)
    image_temp[35, 250] = (255,0,0)
    image_temp[35, 260] = (255,0,0)
    image_temp[25, 220] = (255,0,0)
    image_temp[25, 230] = (255,0,0)
    image_temp[25, 240] = (255,0,0)
    image_temp[25, 250] = (255,0,0)
    image_temp[60, 325] = (255,0,0)
    #/*******************************************************/
    image_temp[60, 240] = (255,0,0) # tree stems and lamps
    image_temp[70, 243] = (255,0,0)
    image_temp[78, 245] = (255,0,0)
    image_temp[78, 336] = (255,0,0)
    image_temp[71, 336] = (255,0,0)
    image_temp[75, 337] = (255,0,0)
    image_temp[73, 337] = (255,0,0)
    image_temp[73, 495] = (255,0,0)
    image_temp[78, 494] = (255,0,0)
    image_temp[73, 543] = (255,0,0)
    image_temp[78, 543] = (255,0,0)

    return image_temp
