'''
Created on Apr 14, 2014

@author: minjoon
'''
from matplotlib import cm
import os
import sys

import cv2

from image.low_level import open_img, BinarizedSegmentation
from image.ocr import LabelSaver, LabelRecognizer
from image.visual_primitive import VPGenerator, VPRecorder
import matplotlib.pyplot as plt


def record():
    recorder = VPRecorder()
    recorder.record(sys.argv[1])
    

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
    imgpath = sys.argv[1]
    folderpath = os.path.dirname(imgpath)
    filepath = os.path.join(folderpath, 'vp_sln_0000.csv')
    img = open_img(sys.argv[1])
    bin_seg = BinarizedSegmentation(img)
    rc = LabelRecognizer()
    for seg in bin_seg.label_seg_list:
        seg.assign_label(rc.recognize(seg.img))

    # line_params:  (rho, theta, line_mg, line_ml, th, nms_rho, nms_theta)
    # circle_params: (dp, minRadius, maxRadius, arc_mg, arc_ml, params1, params2, minDist)
    #line_params = (1,np.pi/180,3,20,30,2,2)
    # circle_params = (1,20,200,3,20,50,50,2)
    #vpg = VPGenerator(bin_seg,eps=1.5)
    vpg = VPGenerator(filepath=filepath)
    out_img = cv2.cvtColor(img,cv2.cv.CV_GRAY2BGR)
    
    for vp in vpg.vpline_list:
        x0,y0,x1,y1 = [int(round(float(elem))) for elem in vp.abs_line_tuple]
        cv2.line(out_img,(x0,y0),(x1,y1),(255,0,0),1)
    for vp in vpg.vparc_list:
        x,y,r,t0,t1 = [int(round(float(elem))) for elem in vp.abs_arc_tuple]
        cv2.circle(out_img,(int(x),int(y)),int(r),(0,255,0),1)
    
    cv2.imshow('detected circles', out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #vpg.save(filepath)
    
    
    
     
if __name__ == '__main__':
    test()