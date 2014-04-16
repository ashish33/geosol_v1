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
from image.visual_primitive import VPGenerator, VPRecorder, display_vp, \
    evaluate_solution
import matplotlib.pyplot as plt
import numpy as np


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
        
def save_solution():
    imgpath = sys.argv[1]
    folderpath = os.path.dirname(imgpath)
    img = open_img(imgpath)
    bin_seg = BinarizedSegmentation(img)
    rc = LabelRecognizer()
    for seg in bin_seg.label_seg_list:
        seg.assign_label(rc.recognize(seg.img))

    
    # line_params:  (rho, theta, line_mg, line_ml, th, nms_rho, nms_theta)
    # circle_params: (dp, minRadius, maxRadius, arc_mg, arc_ml, params1, params2, minDist)
    itr = 1
    digit = 6
    num = 10000
#    for itr in range(num):
    line_params = (1,np.pi/180,3,20,30,2,np.pi/60)
    circle_params = (1,20,100,2,20,50,40,2)
    naivepath = os.path.join(folderpath,'naive')
    infopath = os.path.join(naivepath, 'info.csv')
    info_fh = open(infopath, 'a')
    
    name = ('{0:0%d}' %digit).format(itr) + '.csv'
    while name in os.listdir(naivepath):
        itr += 1
        name = ('{0:0%d}' %digit).format(itr) + '.csv'
    slnpath = os.path.join(naivepath,name)
    #vpg = VPGenerator(bin_seg,eps=1.5)
    vpg = VPGenerator(bin_seg, line_params=line_params, circle_params=circle_params)
    info = "%s,%s,%s\n" %(name,str(line_params),str(circle_params))
    info_fh.write(info)
    out_img = display_vp(img, vpg)
    
    cv2.imshow('detected circles', out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    vpg.save(slnpath)
    info_fh.close()
    
def evaluate():
    folderpath = sys.argv[1]
    img_path = os.path.join(folderpath, 'original.gif')
    img = open_img(img_path)
    true_sln_path = os.path.join(folderpath, 'gt_vp_sln.csv')
    true_vpg = VPGenerator(filepath=true_sln_path)
    test_sln_path = os.path.join(folderpath, 'vp_sln_0000.csv')
    test_vpg = VPGenerator(filepath=test_sln_path)
    out_img = display_vp(img, true_vpg)
    cv2.imshow('test solution', out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print evaluate_solution(test_vpg, true_vpg)
     
if __name__ == '__main__':
    save_solution()