import time

import cv2
# from DB.mongo_connection import get_mongo_client
from loguru import logger
import numpy as np
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fish_img =  cv2.imread('imtest.jpg')



def draw_dt_on_np(im, detections, print_dt=False, color=(255,0,0),
                  text_size=1, **kwargs):
    '''
    im: image numpy array, shape(h,w,3), RGB
    detections: rows of [x,y,w,h,a,conf], angle in degree
    '''
    line_width = kwargs.get('line_width', im.shape[0] // 300)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = max(int(2*text_size), 1)
    #detections
    for bb in detections:
        #bb = bb.tolist()
        if len(bb) == 6:
            x,y,w,h,a,conf = bb
        else:
            x,y,w,h,a = bb[:5]
            conf = -1
        x1, y1 = x - w/2, y - h/2
        if print_dt:
            print(f'[{x} {y} {w} {h} {a}], confidence: {conf}')
        conts = draw_xywha(im, x, y, w, h, a, color=color, linewidth=line_width)
        # if kwargs.get('show_conf', True):
        #     cv2.putText(im, f'{conf:.2f}', (int(x1),int(y1)), font, 1*text_size,
        #                 (255,255,255), font_bold, cv2.LINE_AA)
        # if kwargs.get('show_angle', False):
        #     cv2.putText(im, f'{int(a)}', (x,y), font, 1*text_size,
        #                 (255,255,255), font_bold, cv2.LINE_AA)
    # if kwargs.get('show_count', True):
    #     caption_w = int(im.shape[0] / 4.8)
    #     caption_h = im.shape[0] // 25
    #     start = (im.shape[1] - caption_w, im.shape[0] // 20)
    #     end = (im.shape[1], start[1] + caption_h)
        # cv2.rectangle(im, start, end, color=(0,0,0), thickness=-1)

        # fisheye_rect = (im.shape[1] - caption_w + im.shape[0] // 100, end[1] - im.shape[1] // 200)
        #fisheye_rect = [start,end]
        # cv2.putText(im, f'Count: {len(detections)}',
        #             (im.shape[1] - caption_w + im.shape[0]//100, end[1]-im.shape[1]//200),
        #             font, 1.2*text_size,
        #             (255,255,255), font_bold*2, cv2.LINE_AA)
    return im, conts


def draw_xywha(im, x, y, w, h, angle, color=(0,0,255), linewidth=5):
    '''
    im: image numpy array, shape(h,w,3), RGB
    angle: degree
    '''
    color = (255, 0, 0)
    line_width = im.shape[0] // 300
    c, s = np.cos(angle/180*np.pi), np.sin(angle/180*np.pi)
    R = np.asarray([[c, s], [-s, c]])
    pts = np.asarray([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    rot_pts = []
    for pt in pts:
        rot_pts.append(([x, y] + pt @ R).astype(int))
    contours = np.array([rot_pts[0], rot_pts[1], rot_pts[2], rot_pts[3]])
    cv2.polylines(im, [contours], isClosed=True, color=(0, 0, 255),
                  thickness=2, lineType=cv2.LINE_4)
    return im,contours








