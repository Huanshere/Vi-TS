#!/usr/bin/env python3

import cv2
import numpy as np
import time
import io

dev = 0

cap = cv2.VideoCapture('/dev/video'+str(dev), cv2.CAP_V4L)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)

width = 256
height = 192
scale = 3
newWidth = width*scale 
newHeight = height*scale
alpha = 1.0
rad = 0
threshold = 2

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
        
        lomax = thdata[...,1].max()
        posmax = thdata[...,1].argmax()
        mcol,mrow = divmod(posmax,width)
        himax = thdata[mcol][mrow][0]
        lomax=lomax*256
        maxtemp = himax+lomax
        maxtemp = (maxtemp/64)-273.15
        maxtemp = round(maxtemp,2)
        
        lomin = thdata[...,1].min()
        posmin = thdata[...,1].argmin()
        lcol,lrow = divmod(posmin,width)
        himin = thdata[lcol][lrow][0]
        lomin=lomin*256
        mintemp = himin+lomin
        mintemp = (mintemp/64)-273.15
        mintemp = round(mintemp,2)
        
        loavg = thdata[...,1].mean()
        hiavg = thdata[...,0].mean()
        loavg=loavg*256
        avgtemp = loavg+hiavg
        avgtemp = (avgtemp/64)-273.15
        avgtemp = round(avgtemp,2)
        
        bgr = cv2.cvtColor(imdata,  cv2.COLOR_YUV2BGR_YUYV)
        bgr = cv2.convertScaleAbs(bgr, alpha=alpha)
        bgr = cv2.resize(bgr,(newWidth,newHeight),interpolation=cv2.INTER_CUBIC)
        if rad>0:
            bgr = cv2.blur(bgr,(rad,rad))
        
        heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)
        
        cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
        (int(newWidth/2),int(newHeight/2)-20),(255,255,255),2)
        cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
        (int(newWidth/2)-20,int(newHeight/2)),(255,255,255),2)
        
        cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
        (int(newWidth/2),int(newHeight/2)-20),(0,0,0),1)
        cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
        (int(newWidth/2)-20,int(newHeight/2)),(0,0,0),1)
        
        cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)
        
        if maxtemp > avgtemp+threshold:
            cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,0), 2)
            cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,255), -1)
            cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
            cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)
        
        if mintemp < avgtemp-threshold:
            cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (0,0,0), 2)  
            cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (255,0,0), -1)
            cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
            cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow('Thermal',heatmap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()