from __future__ import absolute_import, division, print_function

import numpy as np

import cv2 as cv

import sys
sys.path.insert(0, 'path/to/caffe')
import caffe

# caffe.set_mode_gpu()

from utils import points_to_box, non_max_suppression

class RSA():
    def __init__(self,
                 net_def1 = 'models/res_pool2.prototxt',
                 net_weights1 = 'models/resnet50.caffemodel',
                 net_def2 = 'models/hm_trans.prototxt',
                 net_weights2 = 'models/hm_trans.caffemodel',
                 net_def3 = 'models/res_3b_s16_f2r.prototxt',
                 net_weights3 = 'models/resnet50.caffemodel',
                 input_scale = 0,
                 scale = (1,2,3,4,5),
                 max_img = 2048,
                 min_img = 64,
                 anchor_scale = 1,
                 factor = 1,
                 anchor_box = (-44.7548,-44.7548,44.7548,44.7548),
                 thresh_cls = 3,
                 stride = 16,
                 anchor_center = 7.5,
                 anchor_pts = (-0.1719,-0.2204,0.1719,-0.2261,-0.0017,-0.0047,-0.1409,0.2034,0.1409,0.1978),
                 nms_thres = 0.2,
                 nms_score = 8
                 ):
        self.input_scale = input_scale
        self.scale = scale
        self.max_img = max_img
        self.min_img = min_img
        self.anchor_scale = anchor_scale

        self.factor = factor
        self.anchor_box = anchor_box
        self.thresh_cls = thresh_cls
        self.stride = stride
        self.anchor_center = anchor_center
        self.anchor_pts = anchor_pts
        self.nms_thres = nms_thres
        self.nms_score = nms_score
        self.net1 = caffe.Net(net_def1, net_weights1, caffe.TEST)
        self.net2 = caffe.Net(net_def2, net_weights2, caffe.TEST)
        self.net3 = caffe.Net(net_def3, net_weights3, caffe.TEST)

    def predict(self, img):
        factor = self.max_img / max(img.shape) * 2**self.input_scale
        img_t = cv.resize(img, (int(round(img.shape[1]*factor)), int(round(img.shape[0]*factor)))) - 127.0
        img_t = img_t.transpose(2,0,1)[np.newaxis,:,:,:]
        self.net1.blobs['data'].reshape(*img_t.shape)
        np.copyto(self.net1.blobs['data'].data, img_t)
        o = self.net1.forward()['res2b'][0]

        scale = np.array(self.scale)
        scale = scale.clip(max=max(self.scale))
        orig_scale = scale.max()
        featmaps = []
        featmaps.append(o)
        sidx = scale[scale < orig_scale][::-1]
        for i in range(sidx.size):
            scale_t = sidx[i]
            if i == 0:
                diffcnt = orig_scale - scale_t
            else:
                diffcnt = sidx[i-1] - scale_t
            inp = featmaps[i][np.newaxis, :, :, :]
            for cnt in range(diffcnt):
                self.net2.blobs['data'].reshape(*inp.shape)
                np.copyto(self.net2.blobs['data'].data, inp)
                o = self.net2.forward()['res2b_trans_5'][0]
                inp = o
            featmaps.append(o.copy())

        scale = np.array(self.scale)
        scale = np.power(2.0, scale[::-1] - 5)
        D = {}
        D['active'] = []
        D['cls_score'] = []
        D['point'] = []
        D['box'] = []
        for i in range(len(featmaps)):
            self.net3.blobs['res2b'].reshape(*featmaps[i][np.newaxis, :, :, :].shape)
            np.copyto(self.net3.blobs['res2b'].data, featmaps[i][np.newaxis, :, :, :])
            o = self.net3.forward()
            pts_out = []
            reg = o['rpn_reg'].squeeze()
            cls = o['rpn_cls'].squeeze()
            anchor_box_len = (self.anchor_box[2] - self.anchor_box[0], self.anchor_box[3] - self.anchor_box[1])
            y, x = np.where(cls >= self.thresh_cls)

            for idx in range(y.size):
                anchor_center_now = ((x[idx])*self.stride + self.anchor_center, (y[idx])*self.stride + self.anchor_center)
                anchor_points_now = np.multiply(self.anchor_pts, anchor_box_len[0]) + np.matlib.repmat(anchor_center_now, 1, 5)
                pts_delta = np.multiply(reg[:,y[idx],x[idx]], anchor_box_len[0])
                pts_out.append(pts_delta + anchor_points_now)
            if cls[y,x].size != 0:
                D['active'].append(cls)
                D['cls_score'].append(cls[y,x])
                D['point'].append(np.array(pts_out).squeeze() / scale[i])
                boxes, failed = points_to_box(np.array(pts_out).squeeze())
                if not failed:
                     boxes /= scale[i]
                else:
                    boxes = np.zeros((len(pts_out), 4))
                D['box'].append(boxes)
        num_faces = 0
        if len(D['cls_score']) != 0:
            D['cls_score'] = np.hstack(D['cls_score'])
            D['point'] = np.vstack(D['point'])
            D['box'] = np.vstack(D['box'])

            img_scale = max(img.shape) / self.max_img
            D['point'] *= img_scale
            D['box'] *= img_scale
            boxes_with_scores = np.hstack([D['box'], np.array(D['cls_score'], ndmin=2).T])
            final_boxes, idx = non_max_suppression(boxes_with_scores, self.nms_thres)
            final_idx = final_boxes[:, 4] > self.nms_score
            num_faces = final_idx.sum()
            return final_boxes[final_idx], D['point'][idx[final_idx]], num_faces
        else:
            no_face = True
            return None, None, num_faces

if __name__ == '__main__':

    rsa = RSA()

    img = cv.imread('testimg2.jpg')
    bboxes, pts, _ = rsa.predict(img)
    bboxes = bboxes.astype(np.int)
    pts = pts.astype(np.int)
    for i in range(bboxes.shape[0]):
        color = (np.random.randint(0, 256),np.random.randint(0, 256),np.random.randint(0, 256))
        cv.rectangle(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]),color,3)
        pt = pts[i].reshape(-1, 2)
        for j in range(pt.shape[0]):
            cv.circle(img, (pt[j,0],pt[j,1]) , 3, color)

    cv.imshow('test', img)

    cv.waitKey()




