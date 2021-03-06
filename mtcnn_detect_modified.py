#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 3:44:55 2018

@author: llk
"""

import _init_paths
import caffe
import cv2
import numpy as np
import sys
import os
import hashlib
from cvk_means import kmeans


def bbreg(boundingbox, reg):
    reg = reg.T

    # calibrate bouding boxes
    if reg.shape[1] == 1:
        print "reshape of reg"
        pass  # reshape of reg
    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1

    bb0 = boundingbox[:, 0] + reg[:, 0]*w
    bb1 = boundingbox[:, 1] + reg[:, 1]*h
    bb2 = boundingbox[:, 2] + reg[:, 2]*w
    bb3 = boundingbox[:, 3] + reg[:, 3]*h

    boundingbox[:, 0:4] = np.array([bb0, bb1, bb2, bb3]).T
    # print "bb", boundingbox
    return boundingbox


def pad(boxesA, w, h):
    boxes = boxesA.copy()

    tmph = boxes[:, 3] - boxes[:, 1] + 1
    tmpw = boxes[:, 2] - boxes[:, 0] + 1
    numbox = boxes.shape[0]

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw
    edy = tmph

    x = boxes[:, 0:1][:, 0]
    y = boxes[:, 1:2][:, 0]
    ex = boxes[:, 2:3][:, 0]
    ey = boxes[:, 3:4][:, 0]

    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
        ex[tmp] = w-1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
        ey[tmp] = h-1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])

    dy = np.maximum(0, dy-1)
    dx = np.maximum(0, dx-1)
    y = np.maximum(0, y-1)
    x = np.maximum(0, x-1)
    edy = np.maximum(0, edy-1)
    edx = np.maximum(0, edx-1)
    ey = np.maximum(0, ey-1)
    ex = np.maximum(0, ex-1)
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]


def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[:, 2] - bboxA[:, 0]
    h = bboxA[:, 3] - bboxA[:, 1]
    l = np.maximum(w, h).T

    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.repeat([l], 2, axis=0).T
    return bboxA


def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: the remaining boxes' index in the `boxes`
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    # sort the boxes in an ascending order w.r.t their scores
    I = np.array(s.argsort())  # read s using I

    pick = []
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        # [xx1, yy1, xx2, yy2] is the overlapped area
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        # leave out all boxes whose IoU > threshold
        I = I[np.where(o <= threshold)[0]]
    return pick


def generateBoundingBox(map, reg, scale, t):
    stride = 2
    '''
     `cellsize` is a different concept from the input shape of the caffe model,
     it determines how large of the 'anchor'(first descirbed in the faster-rcnn).
     But note that it is just a square anchor, as a result, what if
     we deliberately impose several various aspect radio and size anchor on the
     feature map. Would it will detect more proper and accurate faces? Also, one
     Note that the variety of size of anchor has been employed by the image pyramid,
     we need to focus mainly on the various aspect radios.
    '''
    cellsize = 12
    map = map.T
    dx1 = reg[0, :, :].T
    dy1 = reg[1, :, :].T
    dx2 = reg[2, :, :].T
    dy2 = reg[3, :, :].T
    # we only keep the proposal whose class possibility > t
    (x, y) = np.where(map >= t)
    # x, y is the coordinates in the last feature map
    score = map[x, y]
    reg = np.array([dx1[x, y], dy1[x, y], dx2[x, y], dy2[x, y]])

    boundingbox = np.array([y, x]).T
    bb1 = np.fix((stride * boundingbox + 1) / scale).T
    bb2 = np.fix((stride * boundingbox + cellsize) / scale).T
    score = score[np.newaxis, ...]

    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)
    return boundingbox_out.T


def drawBoxes(im, boxes):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    for i in range(x1.shape[0]):
        cv2.rectangle(im, (int(x1[i]), int(y1[i])),
                      (int(x2[i]), int(y2[i])), (0, 255, 0), 1)
    return im


from time import time
_tstart_stack = []


def tic():
    _tstart_stack.append(time())


def toc(fmt="Elapsed: %s s"):
    print fmt % (time()-_tstart_stack.pop())


def detect_face(img, minsize, PNet, RNet, ONet, threshold, fastresize, factor):

    factor_count = 0
    total_boxes = np.zeros((0, 9), np.float)
    points = []
    h, w, _ = img.shape
    minl = min(h, w)
    img = img.astype(float)
    m = 12.0 / minsize
    minl = minl * m

    # create scale pyramid w.r.t the original image
    scales = []
    while minl >= 12:
        scales.append(m * pow(factor, factor_count))
        minl *= factor
        factor_count += 1

    # first stage
    for scale in scales:
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))

        if fastresize:
            im_data = (img - 127.5) * 0.0078125  # [0,255] -> [-1,1]
            im_data = cv2.resize(im_data, (ws, hs))  # default is bilinear
        else:
            im_data = cv2.resize(img, (ws, hs))  # default is bilinear
            im_data = (im_data - 127.5) * 0.0078125  # [0,255] -> [-1,1]

        im_data = np.swapaxes(im_data, 0, 2)
        im_data = im_data[np.newaxis, ...]
        PNet.blobs['data'].reshape(1, 3, ws, hs)
        PNet.blobs['data'].data[...] = im_data
        out = PNet.forward()
        boxes = generateBoundingBox(out['prob1'][0, 1, :, :], out['conv4-2'][0],
                                    scale, threshold[0])
        if boxes.shape[0] != 0:
            pick = nms(boxes, 0.5, 'Union')

            if len(pick) > 0:
                boxes = boxes[pick, :]

        if boxes.shape[0] != 0:
            total_boxes = np.concatenate((total_boxes, boxes), axis=0)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]

        # revise and convert to square
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        t1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        t2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        t3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        t4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        t5 = total_boxes[:, 4]
        total_boxes = np.array([t1, t2, t3, t4, t5]).T

        total_boxes = rerec(total_boxes)  # convert box to square

        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

    numbox = total_boxes.shape[0]

    if numbox > 0:
        # second stage
        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3))  # (24, 24, 3, numbox)
        for k in range(numbox):
            tmp = np.zeros((int(tmph[k]) + 1, int(tmpw[k]) + 1, 3))
            tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k]) +
                1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
            tempimg[k, :, :, :] = cv2.resize(tmp, (24, 24))

        # done in imResample function wrapped by python
        tempimg = (tempimg-127.5)*0.0078125
        # RNet

        tempimg = np.swapaxes(tempimg, 1, 3)

        RNet.blobs['data'].reshape(numbox, 3, 24, 24)
        RNet.blobs['data'].data[...] = tempimg
        out = RNet.forward()

        score = out['prob1'][:, 1]
        pass_t = np.where(score > threshold[1])[0]

        score = np.array([score[pass_t]]).T
        total_boxes = np.concatenate((total_boxes[pass_t, 0:4], score), axis=1)

        mv = out['conv5-2'][pass_t, :].T
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            if len(pick) > 0:
                total_boxes = total_boxes[pick, :]
                total_boxes = bbreg(total_boxes, mv[:, pick])
                total_boxes = rerec(total_boxes)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage

            total_boxes = np.fix(total_boxes)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw,
                tmph] = pad(total_boxes, w, h)

            tempimg = np.zeros((numbox, 48, 48, 3))
            for k in range(numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k]) +
                    1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                tempimg[k, :, :, :] = cv2.resize(tmp, (48, 48))
            tempimg = (tempimg-127.5)*0.0078125  # [0,255] -> [-1,1]

            # ONet
            tempimg = np.swapaxes(tempimg, 1, 3)
            ONet.blobs['data'].reshape(numbox, 3, 48, 48)
            ONet.blobs['data'].data[...] = tempimg
            out = ONet.forward()

            score = out['prob1'][:, 1]
            points = out['conv6-3']
            pass_t = np.where(score > threshold[2])[0]
            points = points[pass_t, :]
            score = np.array([score[pass_t]]).T
            total_boxes = np.concatenate(
                (total_boxes[pass_t, 0:4], score), axis=1)

            mv = out['conv6-2'][pass_t, :].T
            w = total_boxes[:, 3] - total_boxes[:, 1] + 1
            h = total_boxes[:, 2] - total_boxes[:, 0] + 1

            points[:, 0:5] = np.tile(
                w, (5, 1)).T * points[:, 0:5] + np.tile(total_boxes[:, 0], (5, 1)).T - 1
            points[:, 5:10] = np.tile(
                h, (5, 1)).T * points[:, 5:10] + np.tile(total_boxes[:, 1], (5, 1)).T - 1

            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv[:, :])
                pick = nms(total_boxes, 0.7, 'Min')

                if len(pick) > 0:
                    total_boxes = total_boxes[pick, :]
                    points = points[pick, :]

    return total_boxes, points


def detect(img):

    minsize = 20

    caffe_model_path = "./model"

    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    caffe.set_mode_gpu()
    PNet = caffe.Net(caffe_model_path+"/det1.prototxt",
                     caffe_model_path+"/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path+"/det2.prototxt",
                     caffe_model_path+"/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path+"/det3.prototxt",
                     caffe_model_path+"/det3.caffemodel", caffe.TEST)

    img_matlab = img[:, :, ::-1]  # GBR to RGB

    boundingboxes, points = detect_face(
        img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
    # img = drawBoxes(img, boundingboxes)
    # cv2.imshow('FaceDetect', img)
    return boundingboxes


def detect_faces(img):
    img_matlab = img[:, :, ::-1]
    boundingboxes, points = detect_face(
        img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
    return boundingboxes.astype('int64')[:, :-1]


def getpaths(dir):
    ls = os.listdir(dir)
    paths = [os.path.join(dir, f) for f in ls if os.path.splitext(
        f)[1] == '.jpg' or os.path.splitext(f)[1] == '.png']
    return paths

def getmd5(fpath):
    md5file = open(fpath,'rb')
    md5=hashlib.md5(md5file.read()).hexdigest()
    md5file.close()
    return md5

def opdir(): 
    imagedir = os.path.join(os.getcwd(), 'rectify')
    fpaths = getpaths(imagedir)
    for i, f in zip(range(len(fpaths)), fpaths):
        try:
            img = cv2.imread(f)
            boundingboxes = detect(img.copy())
        except Exception as e:
            print(e)
            continue
        x1 = boundingboxes[:, 0]
        y1 = boundingboxes[:, 1]
        x2 = boundingboxes[:, 2]
        y2 = boundingboxes[:, 3]
        # for (j,x,y,xx,yy) in zip(range(len(x1)),x1,y1,x2,y2):
        #     # print(img.shape)
        #     row,col,depth = img.shape
        #     ymid = int((yy+y)/2)
        #     ymid = int(yy-(yy-ymid))
        #     xmid = int((xx+x)/2)
        #     # yy = int((yy-ymid)*1.2+ymid)
        #     # yy = min(row-1,yy)
        #     x = int(xmid-(xmid-x)*1.5)
        #     xx = int((xx-xmid)*1.5+xmid)
        #     x = max(x,0)
        #     xx = min(xx,col-1)
        #     try:
        #         half_face = img[int(ymid):int(yy),int(x):int(xx),:]
        #         import time
        #         # cv2.imwrite("hh{}.png".format(int(time.time())), half_face)
        #         img[int(ymid):int(yy),int(x):int(xx),:]=kmeans(i,half_face)
        #     except Exception as e:
        #         print(e)
        
        img = drawBoxes(img, boundingboxes)
        cv2.imshow('FaceDetect', img)
        cv2.moveWindow('FaceDetect',450,0)
        key = cv2.waitKey(0)
        cv2.imwrite('./afterrectify/{}'.format(os.path.split(f)[1]), img)
        # if key & 0xff == ord('1'):
        #     md5 = getmd5(f)
        #     cv2.imwrite('./aftersrectify/{}.jpg'.format(md5), img)

        # cmd = ''
        # if key & 0xff == ord('1'):
        #     cmd = 'cp \"'+f+'\" ' + r'~/Desktop/good'
        # if key & 0xff == ord('2'):
        #     cmd = 'cp \"'+f+'\" ' + r'~/Desktop/bad'
        # if key & 0xff == ord('3'):
        #     cmd = 'cp \"'+f+'\" ' + r'~/Desktop/special'
        # os.system(cmd)
        # if key == 27:
        #     quit()
        # while (cv2.waitKey(0) & 0xFF == 27):
        #     break
        cv2.destroyAllWindows()
 
def opVideo():
    cap = cv2.VideoCapture('suc3.avi')
    while(cap.isOpened()):  
        ret, img = cap.read()
        if ret!=True:
            break


        boundingboxes = detect(img.copy())
        x1 = boundingboxes[:, 0]
        y1 = boundingboxes[:, 1]
        x2 = boundingboxes[:, 2]
        y2 = boundingboxes[:, 3]
        # for (i,x,y,xx,yy) in zip(range(len(x1)),x1,y1,x2,y2):
        #     try:
        #         half_face = img[int((yy+y)/2):int(yy),int(x):int(xx),:]
        #         img[int((yy+y)/2):int(yy),int(x):int(xx),:]=kmeans(i,half_face)
        #     except Exception as e:
        #         print(e)
        # cv2.imwrite('./after/%d.png' % i, img)
        img = drawBoxes(img, boundingboxes)
        cv2.imshow('FaceDetect', img)


        # cv2.imshow('image', frame)  
        k = cv2.waitKey(0)  
        #q键退出
        if (k & 0xff == ord('q')):  
            break 


def camera():
    video_capture=cv2.VideoCapture(0)
    interval=7
    cnt=0
    while True:
        cnt=(cnt + 1) % interval
        _, img=video_capture.read()
        if cnt % interval == 0:
            img=detect(img)
            img,boundingboxes = detect(img)
            cv2.imshow('FaceDetect', img)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    video_capture.release()


if __name__ == "__main__":
    # detect(sys.argv[1])
    # opdir()
    opVideo()
    # camera()
