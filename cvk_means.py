import numpy as np
import cv2
from findcontours_llk import findContoursAnddfs as findC
import colorsys

def kmeans(index,img):
    im = img.copy()
    cmin = 255
    cmax = 0
    for c in range(3):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                cmin = min(img[i][j][c], cmin)
                cmax = max(img[i][j][c], cmax)
    for c in range(3):    
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j][c] = (img[i][j][c]-cmin)/(cmax-cmin+0.00000000000000000001)*255
    # cv2.imshow("test",img)
    # imgs = cv2.split(img);  
    # for i in range(3):
    #     cv2.equalizeHist(imgs[i], imgs[i]);  
    # img = cv2.merge(imgs, 3);  
    imgxy = np.zeros([img.shape[0], img.shape[1], 5])
    imgxy[:,:,0:3] = img
    kk = 100
    imgxy[:,:,3] = [[float(i/imgxy.shape[0])*kk for j in range(imgxy.shape[1])] for i in range(imgxy.shape[0])]
    imgxy[:,:,4] = [[float(j/imgxy.shape[1])*kk for j in range(imgxy.shape[1])] for i in range(imgxy.shape[0])]
    Z = imgxy.reshape((-1, 5))
    # Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)    
    K = 10
    # print(img.shape)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    from random import random
    center_color = np.array([
        [random()*255, random()*255, random()*255]  for i in range(K)
    ])
    center = np.uint8(center)
    center_color = np.uint8(center_color)
    res = center_color[label.flatten()]
    res2 = res.reshape((img.shape))

   

    label_output = label.reshape(img.shape[0:2])
    cnts,K,center = findC(label_output, center)
    # ret, thresh = cv2.threshold(img, 100, 120, 0)
    # blank, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    from random import random
    center_color = np.array([
        [random()*255, random()*255, random()*255]  for i in range(K)
    ])
    for i in range(K):
        rect = cv2.minAreaRect(cnts[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        lenbox = [(0.00001+(box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2)**0.5,
            (0.00001+(box[2][0]-box[1][0])**2+(box[2][1]-box[1][1])**2)**0.5]
        ratioL = max(lenbox[0]/lenbox[1], lenbox[1]/lenbox[0])
        aera = lenbox[0]*lenbox[1]
        cth = 100
        # print(center[i])
        def check_hsv(x):
            # print(x)
            # print((float(x[2]),float(x[1]),float(x[0])))
            y = colorsys.rgb_to_hsv(float(x[2]),float(x[1]),float(x[0]))
            # print(('hsv',y))
            return y[1]<0.2 and y[2] > 180    #0.2  200
        if  check_hsv(center[i]) and ratioL>1.8 and aera/img.shape[1]/img.shape[0]>0.005:
        # if  check_hsv(center[i]) and aera/img.shape[1]/img.shape[0]>0.01:
            cv2.drawContours(im,[box],0, center_color[i],2)
            print('color',center[i])
            print('Lratio',ratioL)
            print('Sratio',aera/img.shape[1]/img.shape[0])
    # cv2.imwrite('k-means_output.png', res2)
    # cv2.imwrite('contours_output.png', img)
    # cv2.imshow('contour%d'%index,img)
    cv2.imshow('cigarette%d'%index,res2)
    return im
    # cv2.waitKey(0)

if __name__ == '__main__':
    img = cv2.imread('mouth1.jpg')
    kmeans(1, img)
    cv2.waitKey(0)