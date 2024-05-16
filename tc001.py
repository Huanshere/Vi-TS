
import cv2
import numpy as np

dev = 2
cap = cv2.VideoCapture('/dev/video'+str(dev), cv2.CAP_V4L)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
width = 256
height = 192
alpha = 1.0
rad = 0

# 定义颜色映射选项
color_maps = [cv2.COLORMAP_JET, cv2.COLORMAP_COOL, cv2.COLORMAP_GRAY]
color_map_index = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        imdata, thdata = np.array_split(frame, 2)
        hi = thdata[96][128][0]
        lo = thdata[96][128][1]
        lo = lo * 256
        rawtemp = hi + lo
        temp = (rawtemp / 64) - 273.15
        temp = round(temp, 2)
        
        bgr = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
        bgr = cv2.convertScaleAbs(bgr, alpha=alpha)
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_CUBIC)
        
        if rad > 0:
            bgr = cv2.blur(bgr, (rad, rad))
        
        heatmap = cv2.applyColorMap(bgr, color_maps[color_map_index])
        
        # 预先计算坐标
        center_x = int(width / 2)
        center_y = int(height / 2)
        line_length = 20
        # 使用预先计算的坐标绘制线条和文本
        cv2.line(heatmap, (center_x, center_y + line_length), (center_x, center_y - line_length), (255, 255, 255), 2)
        cv2.line(heatmap, (center_x + line_length, center_y), (center_x - line_length, center_y), (255, 255, 255), 2)
        cv2.line(heatmap, (center_x, center_y + line_length), (center_x, center_y - line_length), (0, 0, 0), 1)
        cv2.line(heatmap, (center_x + line_length, center_y), (center_x - line_length, center_y), (0, 0, 0), 1)
        
        cv2.putText(heatmap, str(temp) + ' C', (center_x + 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(heatmap, str(temp) + ' C', (center_x + 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow('Thermal', heatmap)
        
        # 按下 'c' 键切换颜色映射
        if cv2.waitKey(1) & 0xFF == ord('c'):
            color_map_index = (color_map_index + 1) % len(color_maps)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()