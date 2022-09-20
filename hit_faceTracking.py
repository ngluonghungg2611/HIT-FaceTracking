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
retina_face = RetinaFace(model_name='retina_s')
arc_face = ArcFace(model_name='arcface_s')
bt = BYTETracker()
landmark = Landmark()


recog_data = {
    'userID':[],
    'emb':[],
    'trackID':[]
}
database_emb = {
    'userID': [],
    'embs': []
}
img_data_list = os.listdir('data')
for i in range(len(img_data_list)):
    img_path = os.path.join('data', img_data_list[i])
    img = cv2.imread(img_path)
    fbox, kpss = retina_face.detect(img)
    tbox, tids = bt.predict(img, fbox)
    print(kpss[0])
    # face = img[box[1]:box[3],box[0]:box[2]]    
    emb = arc_face.get(img, kpss[0])
    database_emb['embs'].append(emb)
    database_emb['userID'].append(img_data_list[i][:-4])
print('Extract feature on databse done!')

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
        
        for idt, emb, tbox in zip(current_tracking['track_id'], current_tracking['embs'], tboxes):
            box = tbox[:4].astype(int)
            
            dis = np.linalg.norm(database_emb['embs'] - emb, axis = 1)
            if (min(dis) < 25):
                idx = np.argmin(dis)
                t = time.time()
                print(database_emb['userID'][idx])
                # if (time.time() - t > 5):
                cv2.putText(frame, str(database_emb['userID'][idx]), (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
            else:
                t = time.time()
                print("Stranger")
                # if (time.time() - t > 5):
                cv2.putText(frame, "Stranger", (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
                    
        # print(len(current_tracking['embs']))
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