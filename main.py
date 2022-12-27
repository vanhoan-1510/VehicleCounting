import cv2
import torch
import numpy as np
from tracker import *


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture('vehiclevideo.mp4')

tracker = Tracker()


def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)                     


area1 = [(424,514), (666,439), (652,412), (414,469)]
area2 = [(665,425), (827,360), (855,391), (682,461)]

area_1 = set()
area_2 = set()
while True:
    ret,frame=cap.read()
    if not ret:
        break

    frame=cv2.resize(frame,(1020,600))
    results=model(frame)

    list = []

    for index, rows in results.pandas().xyxy[0].iterrows():
        x = int(rows[0])
        y = int(rows[1])
        x1 = int(rows[2])
        y1 = int(rows[3])
        b = str(rows['name'])
        if "car" in b or "motorcycle" in b:
            list.append([x,y,x1,y1])

    idx_bbox = tracker.update(list)

    for bbox in idx_bbox:
        x2, y2, x3, y3, id = bbox

        cv2.rectangle(frame, (x2,y2), (x3,y3), (0,0,255), 2)
        cv2.putText(frame, str(id), (x2,y2), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
        cv2.circle(frame, (x3, y3), 4, (0, 255, 0), -1)

        result_1 = cv2.pointPolygonTest(np.array(area1, np.int32),((x3,y3)), False)
        result_2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x3, y3)), False)

        if result_1 > 0:
            area_1.add(id)

        if result_2 > 0:
            area_2.add(id)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 255), 3)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 255), 3)

    a1 = len(area_1)
    a2 = len(area_2)

    cv2.putText(frame, str(a1), (544,520), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
    cv2.putText(frame, str(a2), (805,445), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
              
    cv2.imshow("FRAME",frame)
    if cv2.waitKey(100)&0xFF==ord('q'):
        break
# cap.release()
# cv2.destroyAllWindows()
