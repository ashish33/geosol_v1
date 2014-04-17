'''
Created on Apr 14, 2014

@author: minjoon
'''
from matplotlib import cm
import os
import sys

import cv2

from external.path import directory_iterator, find_file, next_name
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

def get_random_params():
    th = np.random.random_integers(30,100)
    nms_rho = np.random.random_integers(2,10)
    nms_theta = np.random.random_integers(3*np.pi/180,10*np.pi/180)
    param2 = np.random.random_integers(40,100)
    minDist = np.random.random_integers(2,10)
    line_params = (1,np.pi/180,3,20,th,nms_rho,nms_theta)
    circle_params = (0.5,20,150,2,20,50,param2,minDist)
    return (line_params,circle_params)

def baseline_evaluation(problem_path, num):
    f1_list = []
    for itr in range(num):
        line_params, circle_params = get_random_params()
        num, tp, fp = baseline_test(problem_path, line_params, circle_params)
        p, r = precision_recall(num, tp, fp)
        f1 = float(p*r)/(p+r)
        f1_list.append(f1)
        
    dir_list = directory_iterator(problem_path, 4)
    problem_size = len(dir_list)
    title = 'P=%d, N=%d' %(problem_size, num)
    img_path = next_name(problem_path)
    plt.hist(f1_list)
    plt.title(title)
    plt.savefig()
        

'''
problem_path: the path to the root
returns f1 value
'''
        
def baseline_test(problem_path, line_params, circle_params):
    dir_list = directory_iterator(problem_path, 4)
    total = np.array([0,0,0])
    for directory in dir_list:
        img_path = find_file(directory, 'original')
        gt_sln_path = os.path.join(directory, 'gt_vp_sln.csv')
        sln_dir = os.path.join(directory, 'naive')
        if not os.path.exists(sln_dir):
            os.makedirs(sln_dir)
        
        img = open_img(img_path)
        bin_seg = BinarizedSegmentation(img)
        vpg = VPGenerator(bin_seg, line_params=line_params, circle_params=circle_params)
        gt_vpg = VPGenerator(filepath=gt_sln_path)
        
        curr = np.array(evaluate_solution(vpg, gt_vpg))
        total += curr
        vpg.save(next_name(sln_dir,6,'csv'), img)
        
    result_path = os.path.join(problem_path, 'result.csv')
    fh = open(result_path, 'a')
    line_str = ';'.join([str(x) for x in line_params])
    circle_str = ';'.join([str(x) for x in circle_params])
    fh.write('hough,%d,%d,%d,%s,%s\n' %(int(total[0]),int(total[1]),int(total[2]),line_str,circle_str))
    fh.close()
    return total

def precision_recall(num, tp, fp):        
    recall = 0
    if num > 0:
        recall = float(tp)/num
    precision = 0
    if tp+fp > 0:
        precision = float(tp)/(tp+fp)
    return (precision, recall)
    
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
    baseline_evaluation(sys.argv[1], 100)