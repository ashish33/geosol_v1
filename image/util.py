'''
Created on Apr 30, 2014

@author: minjoon
'''
import numpy as np
import cv2

def draw_line(img, line, color, width):
    x0,y0,x1,y1 = [int(np.around(float(elem))) for elem in line]
    cv2.line(img,(x0,y0),(x1,y1),color,width)
    
def draw_arc(img, circle, color, width):
    x,y,r,t0,t1 = [int(np.around(float(elem))) for elem in arc]
    cv2.circle(img,(int(x),int(y)),int(r),color,width)

