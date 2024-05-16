#!/usr/bin/env python3
import cv2
import numpy as np

dev = 0
cap = cv2.VideoCapture('/dev/video'+str(dev), cv2.CAP_V4L)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
width = 256
height = 192
alpha = 1.0
rad = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        imdata,thdata = np.array_split(frame, 2)
        hi = thdata[96][128][0]
        lo = thdata[96][128][1]
        lo = lo*256
        rawtemp = hi+lo
        temp = (rawtemp/64)-273.15
        temp = round(temp,2)
        
        bgr = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
        bgr = cv2.convertScaleAbs(bgr, alpha=alpha)
        bgr = cv2.resize(bgr,(width,height),interpolation=cv2.INTER_CUBIC)
        if rad>0:
            bgr = cv2.blur(bgr,(rad,rad))
        heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)
        cv2.line(heatmap,(int(width/2),int(height/2)+20),\
                 (int(width/2),int(height/2)-20),(255,255,255),2)
        cv2.line(heatmap,(int(width/2)+20,int(height/2)),\
                 (int(width/2)-20,int(height/2)),(255,255,255),2)
        cv2.line(heatmap,(int(width/2),int(height/2)+20),\
                 (int(width/2),int(height/2)-20),(0,0,0),1)
        cv2.line(heatmap,(int(width/2)+20,int(height/2)),\
                 (int(width/2)-20,int(height/2)),(0,0,0),1)
        cv2.putText(heatmap,str(temp)+' C', (int(width/2)+10, int(height/2)-10),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(heatmap,str(temp)+' C', (int(width/2)+10, int(height/2)-10),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow('Thermal',heatmap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()