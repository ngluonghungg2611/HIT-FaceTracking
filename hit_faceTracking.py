import cv2
import numpy as np
import time
from hawk_eyes.face import RetinaFace, ArcFace, Landmark
from hawk_eyes.tracking import BYTETracker
from collections import defaultdict 
import threading
import argparse
import math
import os
def def_value(): 
    return "_"

# app = Flask(__name__)
retina_face = RetinaFace(model_name='retina_m')
arc_face = ArcFace(model_name='arcface_m')
bt = BYTETracker()
landmark = Landmark()
gamma = 37.7952755906
recog_data = {
    'userID':[],
    'emb':[],
    'trackID':[]
}
track_name = defaultdict(def_value)
track_emb = {}
current_tracking = {}
name_idx = 0


cap = cv2.VideoCapture(0)


def recog():
    global track_emb, track_name, recog_data, current_tracking
    ret, _ = cap.read()
    t = time.time()
    
    while ret:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        fboxes,kpss = retina_face.detect(frame)
        tboxes, tids = bt.predict(frame, fboxes)
        tkpss = [None]*len(fboxes)
        for i in range(len(tboxes)):
            min_d = 9e5
            tb = tboxes[i]
            for j in range(len(fboxes)):
                fb = fboxes[j]
                d = abs(tb[0]-fb[0])+abs(tb[1]-fb[1])+abs(tb[2]-fb[2])+abs(tb[3]-fb[3])
                if d < min_d:
                    min_d = d
                    tkpss[i] = kpss[j]
        embs = []
        ids = []
        for tid, tbox, tkps in zip(tids, tboxes, tkpss):
            box = tbox[:4].astype(int)
            land = landmark.get(frame, tbox)
            angle = landmark.get_face_angle(frame, land, False)[1]
            # st = recog_data['userID'][recog_data['trackID'].index(tid)]
            if abs(angle) < 15:
                
                emb = arc_face.get(frame, tkps)
                embs.append(emb)
                ids.append(tid)
            # cv2.putText(frame, st, (box[0], box[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            draw_fancy_box(frame, (box[0],box[1]), (box[2],box[3]), (127, 255, 255), 2, 10, 20)
        # print('embs: ', embs)
        # print('ids: ', ids)
        
        current_tracking = {'frame':frame.copy(), 'track_id': ids, 'embs':embs} 
        
        print(len(current_tracking['embs']))
        # print(current_tracking)
        cv2.imshow('qwe', frame)
        cv2.waitKey(1)
        
def draw_fancy_box(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
        
if __name__ == '__main__':
    recog()