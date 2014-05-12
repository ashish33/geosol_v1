'''
Created on Apr 30, 2014

@author: minjoon
'''
import numpy as np
import cv2

def draw_line(img, line, color, width):
    x0,y0,x1,y1 = [int(np.around(float(elem))) for elem in line]
    cv2.line(img,(x0,y0),(x1,y1),color,width)
    
def draw_arc(img, arc, color, width):
    x,y,r,t0,t1 = arc
    if t0 == 0 and t1 == 0:
        t1 = 2*np.pi
    cv2.ellipse(img,(int(x+0.5),int(y+0.5)),(int(r+0.5),int(r+0.5)), \
                0,t0*180/np.pi,t1*180/np.pi,color,width)
    