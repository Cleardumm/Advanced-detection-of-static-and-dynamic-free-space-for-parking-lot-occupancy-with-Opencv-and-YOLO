import cv2
import numpy as np
import json
import os

#classify two categories manually
def classify(path_ofjson,nameofimg,pathofimg):

    #read the img from path
    img = cv2.imread(pathofimg)

    #open josn file
    with open(path_ofjson, 'r') as f:
        josn_file = json.load(f)

    #from dictionary get 'shapes'
    dir = josn_file['shapes']

    #get through all the points in the josnfile
    length = len(dir)
    for i in range(length):

        #get four vertices of a quad
        topleft = dir[i]['points'][0]
        bottomleft = dir[i]['points'][1]
        topright = dir[i]['points'][3]
        bottomright = dir[i]['points'][2]

        #get the Roi from quadrilateral to rectangle
        pts1 = np.float32([topleft, topright, bottomleft, bottomright])
        pts2 = np.float32([[0, 0], [254, 0], [0, 400], [254, 400]])

        #transform matrix
        M = cv2.getPerspectiveTransform(pts1, pts2)

        #get output image
        dst = cv2.warpPerspective(img, M, (254, 400))

        #use keyboard for classfy
        cv2.imshow("changed", dst)
        k = cv2.waitKey(0) & 0xFF

        # Identify if 'e' or 'E' is pressed, put the picture into Empty
        if (k == 101 or k == 69):
            cv2.imwrite("../RAW_data/Empty/" + str(i) + nameofimg, dst)
            cv2.destroyAllWindows()

        # Identify if 'z' or 'Z' is pressed, put the picture into Full
        if (k == 102 or k == 70):
            cv2.imwrite("../RAW_data/Full/" + str(i) + nameofimg, dst)
            cv2.destroyAllWindows()


#path of josn
pathofjson = "../RAW_data/josn_file_foto/c5001.json"

#path of images
path = "../RAW_data/images"

#images name
dir = os.listdir(path)
for d in dir:
    path_img = path + "/" + d
    classify(pathofjson, d, path_img)





