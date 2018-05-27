import cv2
import numpy as np

bgr = cv2.imread("ko.png")
bgr = bgr.astype(np.float32)

B = np.mean(bgr[:,:,0])
G = np.mean(bgr[:,:,1])
R = np.mean(bgr[:,:,2])

KB = (R + G + B) / (3 * B);  
KG = (R + G + B) / (3 * G);  
KR = (R + G + B) / (3 * R);  

bgr[:,:,0] = bgr[:,:,0] * KB
bgr[:,:,1] = bgr[:,:,1] * KG
bgr[:,:,2] = bgr[:,:,2] * KR
bgr = np.where(bgr>255, 255, bgr)
bgr = bgr.astype(np.uint8)
cv2.imwrite("whitebalance.png",bgr)

hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
print(hsv.max(axis=(0,1)))
im1 = hsv[:,:,0]<=0.6*180
im2 = hsv[:,:,1]<=0.2*255
im3 = hsv[:,:,2]>=0.6*255
cv2.imwrite("hsv.png",hsv)
i = np.array(im1&im2& im3)*255.

cv2.imwrite("hsv_s.png",i)