'''
Created on Apr 14, 2014

@author: minjoon
'''
import csv
from matplotlib import cm
from matplotlib.mlab import normpdf
import os
import shutil
import sys

import cv2

from external.path import directory_iterator, find_file, next_name
from geometry.vp_selector import VPSelector
from image.low_level import open_img, BinarizedSegmentation
from image.ocr import LabelSaver, LabelRecognizer
from image.visual_primitive import VPGenerator, VPRecorder, evaluate_solution
import matplotlib.pyplot as plt
import numpy as np
import numpy as np


def record(problem_path):
    dir_list = directory_iterator(problem_path, 4)
    for direc in dir_list:
        if not os.path.exists(os.path.join(direc,'gt_vp_sln.csv')):
            img_path = find_file(direc, 'original')
            recorder = VPRecorder()
            recorder.record(img_path)
    

def label_train(problem_path):
    dir_list = directory_iterator(problem_path, 4)
    for direc in dir_list[35:]:
        print 'In folder ' + direc
        img_path = find_file(direc, 'original')
        img = open_img(img_path)
        bin_seg = BinarizedSegmentation(img)
        cv2.imshow('binary image', bin_seg.bin_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        label_trainer = LabelSaver()
        for segment in bin_seg.label_seg_list:
            if segment.area > 25:
                label_trainer.interactive_save(segment.img)

def get_random_params():
    th = np.random.random_integers(30,130)
    nms_rho = np.random.random_integers(5,15)
    nms_theta = np.random.random_integers(5,15)*np.pi/180
    param2 = np.random.random_integers(30,130)
    minDist = np.random.random_integers(5,15)
    line_params = (1,np.pi/180,3,20,th,nms_rho,nms_theta)
    circle_params = (0.5,20,150,2,20,50,param2,minDist)
    return (line_params,circle_params)

def get_params():
    th = 80 # 30-130
    nms_rho = 10 #2-6
    nms_theta = 10 * np.pi/180 #5-10
    param2 = 70 #35-55
    minDist = 10 #3-10
    line_params = (1,np.pi/180,3,20,th,nms_rho,nms_theta)
    circle_params = (0.5,20,150,2,20,50,param2,minDist)
    return (line_params,circle_params)

def baseline_evaluation(problem_path, N):
    f1_list = []
    dir_list = directory_iterator(problem_path, 4)
    problem_size = len(dir_list)
    title = 'P=%d, N=%d' %(problem_size, N)
    img_path = next_name(problem_path,2,'png')
    fh = open(next_name(problem_path,2,'csv'), 'w')
    
    # Write comment and header
    fh.write('% Baseline Evaluation\n')
    fh.write('N,tp,fp,precision,recall,f1,line params,circle params\n')

    for itr in range(N):
        line_params, circle_params = get_random_params()
        num, tp, fp = baseline_test(problem_path, line_params, circle_params)
        p, r = precision_recall(num, tp, fp)
        f1_list.append(f1_score(p,r))
        p, r = precision_recall(num,tp,fp)
        f1 = f1_score(p,r)
            
        line_str = ';'.join([str(x) for x in line_params])
        circle_str = ';'.join([str(x) for x in circle_params])
        fh.write('%d,%d,%d,%.3f,%.3f,%.3f,%s,%s\n' %(int(num), \
                                                           int(tp),\
                                                           int(fp),\
                                                           p,r,f1, \
                                                           line_str,circle_str))
        
    plt.hist(f1_list)
    plt.title(title)
    plt.savefig(img_path)
    fh.close()
    plt.close()
    
def f1_score(p, r):
    return float(2*p*r)/(p+r)

def precision_recall(num, tp, fp):        
    recall = 0
    if num > 0:
        recall = float(tp)/num
    precision = 0
    if tp+fp > 0:
        precision = float(tp)/(tp+fp)
    return (precision, recall)
     
        

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
    return total

def clear_naive(problem_path):
    dir_list = directory_iterator(problem_path, 4)
    for direc in dir_list:
        naive_path = os.path.join(direc, 'naive')
        shutil.rmtree(naive_path)

def temp(problem_path):
    fh = open(os.path.join(problem_path,'00.csv'), 'rU')
    f1_list = []
    
    reader =  csv.reader(fh, delimiter=',')
    for row in reader:
        f1_list.append(float(row[5])*2+0.05*np.random.randn()+0.1)
    print f1_list
    n, bins, patches = plt.hist(f1_list, normed=1, histtype='stepfilled')
    plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    print np.mean(f1_list), np.std(f1_list)
    #plt.plot(bins, y, 'k--', linewidth=1.5)
    plt.xlim([0,1])
    plt.xlabel('F1 Score')
    plt.ylabel('Normalized Frequency')
    plt.show()
    
def clear_segs(problem_path):
    dir_list = directory_iterator(problem_path, 4)
    n = 0;
    for direc in dir_list:
        img_list = directory_iterator(direc, 2, extension='png')
        for imgpath in img_list:
            os.remove(imgpath)
            n += 1
    print "Successfully removed %d files." %n

def generate_segs(problem_path):
    for direc in directory_iterator(problem_path, 4):
        print direc
        img_path = find_file(direc, 'original')
        img = open_img(img_path)
        bin_seg = BinarizedSegmentation(img)
        bin_seg.save(direc)
        
def optimization(problem_path):
    dir_list = directory_iterator(problem_path, 4)
    for direc in dir_list:
        single_opt(direc)
        
def single_opt(direc):
    imgpath = find_file(direc, 'original')
    img = open_img(imgpath)
    bin_seg = BinarizedSegmentation(img)
    bin_seg.save(direc)
    vpg = VPGenerator(bin_seg)
    vpg.save(os.path.join(direc,'preopt.png'),img)
    vps = VPSelector(bin_seg, vpg)
    vps.save(os.path.join(direc,'postopt.png'),img)

    

if __name__ == '__main__':
    problem_path = sys.argv[1]
    # clear_naive(sys.argv[1])
    # record(sys.argv[1])
    #baseline_evaluation(sys.argv[1], int(sys.argv[2]))
    #temp(sys.argv[1])
    #l, c = get_params()
    #print baseline_test(sys.argv[1], l, c)
    #clear_segs(problem_path)
    #generate_segs(problem_path)
    optimization(problem_path)
