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
            kernel = np.ones((3,3), np.uint8)
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

def label_resize(img, target_len=16):
    ylen, xlen = np.shape(img)
    if ylen > xlen:
        new_ylen = target_len
        new_xlen = int(round(float(xlen*target_len)/ylen))
    else:
        new_xlen = target_len
        new_ylen = int(round(float(ylen*target_len)/xlen))
    
    target_img = np.ones((target_len,target_len))*255
    resized_img = cv2.resize(img, (new_xlen,new_ylen))
    target_img[:new_ylen,:new_xlen] = resized_img
    return target_img

'''
Given a list of label segments, train OCR
This is to be used by LabelRecognizer
For now, just saves the labels and user needs to
manually modify the csv file
'''
class LabelSaver:
    def __init__(self, path='data/ocr/', digit=6, data_filename='data.csv', img_len=16):
        self.path = path
        self.digit = digit
        self.data_fh = open(path+data_filename, 'ar')
        self.img_len = img_len
        
        self.num = 1
    
    def interactive_save(self, img):
        name = self.next_name()
        cv2.imwrite(self.path+name, img)
        char = raw_input("Label for %s: " %name)
        self.data_fh.write("%s,%s," %(name,char))
        flat_img = label_resize(img).flatten()
        self.data_fh.write(",".join([str(int(e)) for e in flat_img])+'\n')
        
        
    def save(self, img, char): 
        # save segment to next_name
        name = self.next_name()
        cv2.imwrite(self.path+name, img)
        self.csv_fh.write('%s,%s' %(name,char))
    
    def next_name(self):
        name = ('{0:0%d}' %self.digit).format(self.num) + '.png'
        while name in os.listdir(self.path):
            self.num += 1
            name = ('{0:0%d}' %self.digit).format(self.num) + '.png'
        return name
    
    '''
    label outstanding files via interactive mode
    '''
    def label_files(self):
        pass
    
'''
'''
class LabelRecognizer:
    def __init__(self, path='data/ocr/', digit=6, data_filename='data.csv', img_len=16):
        self.path =path
        self.digit = digit
        self.feature_list = []
        self.char_list = []
        self.img_len = img_len
        
        # initialize feature dictionary
        data_fh = open(path+data_filename, 'r')
        reader = csv.reader(data_fh, delimiter=',')
        for row in reader:
            self.char_list.append(row[1])
            self.feature_list.append([int(e) for e in row[2:]])
            
        
    def recognize(self, img):
        new_img = label_resize(img, target_len=self.img_len) 
        flat_img = new_img.flatten()
        cost_list = [np.linalg.norm(np.array(feature)-flat_img) for feature in self.feature_list]
        idx = np.argmin(cost_list)
        return self.char_list[idx]
        
    
    
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
        label_trainer.interactive_label(segment.img)
        
def test():
    img = open_img(sys.argv[1])
    bin_seg = BinarizedSegmentation(img)
    label_recog = LabelRecognizer()
    for segment in bin_seg.label_seg_list:
        print label_recog.recognize(segment.img)
    
if __name__ == '__main__':
    train()
    
    
