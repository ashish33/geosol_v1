'''
Created on Apr 10, 2014

@author: minjoon
'''
'''
Given fiepath, creates cv2 image matrix.
If the filepath is not a compatible file type ('png'),
convert the original image to the compatible type then read it.
'''

import os.path
from scipy import ndimage
import sys

from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import xml.etree.ElementTree as ET
import csv

def open_img(filepath, filetype='png', color='GRAY'):
    newpath = '%s.%s' %(filepath.split('.')[0],filetype)
    if not os.path.isfile(newpath):
        Image.open(filepath).save(newpath)
    img = cv2.imread(newpath)
    if color == 'GRAY':
        return cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
    else:
        return img

def binarize(img, alg='otsu'):
    '''
    Otsu's method for image binarization
    '''
    if alg == 'otsu':
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return thresh
    
def inverse_img(img):
    return 255 - img

'''
Image multiplication
Given img and bool_mat, for all False element set 255.
'''
def bool_img(img, bool_mat):
    return inverse_img(inverse_img(img)*bool_mat)

class ImageSegment:
    def __init__(self, img, loc):
        self.img = img
        self.loc = loc
        self.size = np.shape(img)
        self.slice_img = None # To be used for noisy slice
        
    def set_slice_img(self, img):
        self.slice_img = img

    
'''
img: the image to be segmented
algorithm: algorithm to use for segmentation. Defalut: naive way, 8 side nbr

Segmentation object has
segment_list: entire list of segments
get_diagram(algorithm='largest'): returns the diagram segment.
get_label_list()

'''
class BinarizedSegmentation:
    def __init__(self, img, bin_alg='otsu', seg_alg='naive', dgm_crit='area'):
        bin_img = binarize(img, alg=bin_alg)
        segment_list = []
        
        if seg_alg == 'naive':
            s = [[1,1,1],[1,1,1],[1,1,1]]
            labeled, nr_objects = ndimage.label(bin_img)
            slices = ndimage.find_objects(labeled)
            for idx, s in enumerate(slices):
                seg_img = bool_img(img[s], (labeled[s]==(idx+1)))
                loc = (s[0].start,s[1].start)
                segment = ImageSegment(seg_img, loc)
                segment.set_slice_img(img[s])
                segment_list.append(segment)
                
        if dgm_crit == 'area':
            largest_idx = -1
            largest_area = 0
            for idx, segment in enumerate(segment_list):
                area = segment.size[0]*segment.size[1]
                if area > largest_area:
                    largest_idx = idx
                    largest_area = area
            self.dgm_seg = segment_list[largest_idx]
            del segment_list[largest_idx]
            self.label_seg_list = segment_list

'''
Given a list of label segments, train OCR
This is to be used by LabelRecognizer
For now, just saves the labels and user needs to
manually modify the csv file
'''
class LabelTrainer:
    def __init__(self, path='data/ocr/', digit=6, csv_name='label.csv'):
        self.path = path
        self.digit = digit
        self.csv_fh = open(path+csv_name, 'a')
        
        self.num = 1
    
    def interactive_train(self, segment):
        name = self.next_name()
        cv2.imwrite(self.path+name, segment.img)
        char = raw_input("Label for %s: " %name)
        self.csv_fh.write("%s,%s\n" %(name,char))
    
    def train(self, segment, char): 
        # save segment to next_name
        name = self.next_name()
        cv2.imwrite(self.path+name, segment.img)
        self.csv_fh.write('%s,%s' %(name,char))
    
    def next_name(self):
        name = ('{0:0%d}' %self.digit).format(self.num) + '.png'
        while name in os.listdir(self.path):
            self.num += 1
            name = ('{0:0%d}' %self.digit).format(self.num) + '.png'
        return name
    
    
    
def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = open_img(sys.argv[1])
    bin_seg = BinarizedSegmentation(img)
    ax.imshow(bin_seg.dgm_seg.img, cmap=cm.Greys_r)
    plt.show()
    '''
    label_trainer = LabelTrainer()
    for segment in bin_seg.label_seg_list:
        label_trainer.interactive_train(segment)
    '''
    
if __name__ == '__main__':
    main()
    
    
