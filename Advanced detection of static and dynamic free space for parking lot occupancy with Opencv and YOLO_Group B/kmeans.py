import numpy as np
import cv2 as cv

def imshow(winname,img):

    cv.imshow(winname,img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def kemeans_roi(img):

    # choose the midfreespace manullay
    img1 = img[285:305, 0:1279]

    # print(img1.shape)

    # gaussian blur it's a spatial operation does't effect color
    img_gs = cv.GaussianBlur(img1, (15, 15), 0)

    # first change the color space from BGR to HSV
    img_cvt = cv.cvtColor(img_gs, cv.COLOR_BGR2HSV)

    # change img_cvt in shape(?,3)
    Z = img_cvt[:, :, 0].reshape((-1, 1))

    # convert to np.float32 because kmeans in opencv use data type of float
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # define cluster number
    K = 2

    # use opencv kmeans
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # choose the color belongs to freespace
    index = []
    for i in range(K):

        # set the threshold for centern
        if (center[i, 0] <= 50):
            index.append(i)


    # initialize mask
    mask_out = np.zeros((img_cvt.shape[0] * img_cvt.shape[1], 1), dtype=np.uint8)

    # get mask from index
    for i in range(len(index)):
        mask = np.zeros((img_cvt.shape[0] * img_cvt.shape[1], 1), dtype=np.uint8)
        loc = np.where(label.flatten() == index[i])
        mask[loc] = 255
        mask_out += mask

    mask_out = np.reshape(mask_out, (img_cvt.shape[0], img_cvt.shape[1]))

    # inverse mask
    img_inv = 255 - mask_out

    # erode
    kernel = np.ones((3, 3), np.uint8)
    mask_out = cv.dilate(img_inv, kernel, iterations=5)

    # use canny to detect edges
    canny = cv.Canny(mask_out, 30, 200)

    # choose one line from canny ,use this line to judge ,whether it is rightline or leftline
    sample = canny[13, :]
    mask_out_sample = mask_out[13, :]
    index = np.where(sample > 128)
    x_1 = np.array(index[0])

    if(len(x_1)>0):
        dif = np.diff(x_1)

        # remove the line close to each other
        dif = np.array(dif)
        deletidx = []
        for i in range(len(dif)):
            if (dif[i] < 10):
                deletidx.append(i)
        x_1 = np.delete(x_1, deletidx)

        car_rightline = []
        car_leftline = []

        for i in range(len(x_1)):

            if (mask_out_sample[x_1[i] - 1] > 128):
                car_rightline.append(x_1[i])
            else:
                car_leftline.append((x_1[i]))
        print("leftline ", car_leftline)
        print("rightline ", car_rightline)

        freespacerange = []

        if (car_leftline[0] < car_rightline[0]):
            freespacerange.append([0, car_leftline[0]])
            car_leftline = np.delete(car_leftline, 0)
        if (car_rightline[-1] > car_leftline[-1]):
            freespacerange.append([car_rightline[-1], len(sample)])
            car_rightline = np.delete(car_rightline, len(car_rightline) - 1)
        for i in range(len(car_rightline)):
            loc = np.where(car_leftline > car_rightline[i])[0]
            freespacerange.append([car_rightline[i], car_leftline[loc[0]]])

    return freespacerange