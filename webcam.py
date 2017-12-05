from __future__ import absolute_import, division, print_function

import numpy as np

import cv2 as cv

import sys
sys.path.insert(0, 'path/to/caffe')
import caffe
caffe.set_mode_gpu()

from RSA import RSA

if __name__ == '__main__':

    rsa = RSA(max_img=1024)

    cap = cv.VideoCapture(0)
    cnt = 0
    while True:
        ret, frame = cap.read()
        bboxes, pts, num_faces = rsa.predict(frame)
        if num_faces > 0:
            bboxes = bboxes.astype(np.int)
            for i in range(bboxes.shape[0]):
                cv.rectangle(frame,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]),(0,255,0),3)
            pts = pts.astype(np.int)
            for i in range(pts.shape[0]):
                i = pts[i].reshape(-1, 2)
                for j in range(i.shape[0]):
                    cv.circle(frame, (i[j][0],i[j][1]) , 8, (0,0,255), 4)

        cv.putText(frame, 'face/s detected: ' + str(num_faces), (25, 25), cv.FONT_HERSHEY_PLAIN, 2, (255,255,255))
        cv.imshow('demo', frame)
        cv.imwrite('teaser/' + str(cnt).zfill(5) + '.png', frame)
        cnt += 1
        if cv.waitKey(1) & 0xFF == ord('q'): break
