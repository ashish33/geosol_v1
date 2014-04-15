'''
Created on Apr 14, 2014

@author: minjoon
'''
from matplotlib import cm
import sys

import cv2

from image.low_level import open_img, BinarizedSegmentation
from image.ocr import LabelSaver, LabelRecognizer
from image.visual_primitive import VPGenerator
import matplotlib.pyplot as plt
import numpy as np


def train():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = open_img(sys.argv[1])
    bin_seg = BinarizedSegmentation(img)
    
    '''
    ax.imshow(bin_seg.dgm_seg.img, cmap=cm.Greys_r)
    plt.show()
    '''
    label_trainer = LabelSaver()
    for segment in bin_seg.label_seg_list:
        label_trainer.interactive_save(segment.img)
        
def test():
    img = open_img(sys.argv[1])
    bin_seg = BinarizedSegmentation(img)
    label_recog = LabelRecognizer()
    for segment in bin_seg.label_seg_list:
        print label_recog.recognize(segment.img)
        
    ret, thresh = cv2.threshold(bin_seg.dgm_seg.img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(thresh,1,np.pi/180,10,minLineLength,maxLineGap)
    print len(lines[0])
    for x1,y1,x2,y2 in lines[0]:
        plt.imshow(thresh, cmap=cm.Greys_r)
        plt.plot([x1,x2],[y1,y2],'r')
        plt.show()
        
def test2():
    img = open_img(sys.argv[1])
    bin_seg = BinarizedSegmentation(img)
    # line_params:  (rho, theta, line_mg, line_ml, th, nms_rho, nms_theta)
    # circle_params: (dp, minRadius, maxRadius, arc_mg, arc_ml, params1, params2, minDist)
    #line_params = (1,np.pi/180,3,20,30,2,2)
    # circle_params = (1,20,200,3,20,50,50,2)
    vpg = VPGenerator(bin_seg.dgm_seg,eps=1.5)
    vp_list = vpg.get_vp_list()
    print len(vp_list)
    for vp in vp_list:
        plt.plot(([vp.line_tuple[0],vp.line_tuple[2]]),[vp.line_tuple[1],vp.line_tuple[3]])
    plt.show()

    
     
if __name__ == '__main__':
    test2()