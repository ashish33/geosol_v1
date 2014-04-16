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
    truepath = os.path.join(folderpath, 'gt_vp_sln.csv')
    true_vpg = VPGenerator(filepath=truepath)
    img = open_img(imgpath)
    bin_seg = BinarizedSegmentation(img)
    rc = LabelRecognizer()
    for seg in bin_seg.label_seg_list:
        seg.assign_label(rc.recognize(seg.img))

    
    # line_params:  (rho, theta, line_mg, line_ml, th, nms_rho, nms_theta)
    # circle_params: (dp, minRadius, maxRadius, arc_mg, arc_ml, params1, params2, minDist)
    # th 30-100 increment of 1, nms_rho 1-10, nms_theta 3-10 np.pi/180
    # params2 50-100, minDist 2-10
    naivepath = os.path.join(folderpath,'naive')
    infopath = os.path.join(naivepath, 'info.csv')
    info_fh = open(infopath, 'a')
    itr = 1
    digit = 6
    num = 100
    for itr in range(num):
        th = np.random.random_integers(30,100)
        nms_rho = np.random.random_integers(2,10)
        nms_theta = np.random.random_integers(3*np.pi/180,10*np.pi/180)
        param2 = np.random.random_integers(40,100)
        minDist = np.random.random_integers(2,10)
        line_params = (1,np.pi/180,3,20,th,nms_rho,nms_theta)
        circle_params = (0.5,20,150,2,20,50,param2,minDist)
        
        name = ('{0:0%d}' %digit).format(itr) + '.csv'
        while name in os.listdir(naivepath):
            itr += 1
            name = ('{0:0%d}' %digit).format(itr) + '.csv'
        slnpath = os.path.join(naivepath,name)
        #vpg = VPGenerator(bin_seg,eps=1.5)
        vpg = VPGenerator(bin_seg, line_params=line_params, circle_params=circle_params)
        vpg.save(slnpath)
        info = "%s,%s,%s\n" %(name,str(line_params),str(circle_params))
        info_fh.write(info)    
        all, tp, fp = evaluate_solution(vpg, true_vpg)
        recall = 0
        if all > 0:
            recall = float(tp)/all
        precision = 0
        if tp+fp > 0:
            precision = float(tp)/(tp+fp)
        print recall, precision
        out_img = display_vp(img, vpg)
        cv2.imshow('solution', out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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