'''
Created on Apr 10, 2014

@author: minjoon
'''
'''
Given fiepath, creates cv2 image matrix.
If the filepath is not a compatible file type ('png'),
convert the original image to the compatible type then read it.
'''

import os
from scipy import ndimage

from PIL import Image
import cv2

from external.path import next_name
import numpy as np


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
        blur = cv2.GaussianBlur(img, (0,0), 0.7)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
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
    def __init__(self, img, loc, bin_img=None, slice_img=None):
        self.img = img
        self.loc = loc
        self.size = np.shape(img)
        self.slice_img = slice_img # To be used for noisy slice
        self.bin_img = bin_img # binary image; for hough transform
        self.nz_pts = np.transpose(np.nonzero(np.transpose(bin_img)>0))
        self.label = ''
        self.center = np.array(loc) + np.array(self.size)/2.0
        self.area = self.size[0]*self.size[1]
        
    def assign_label(self, label):
        self.label = label
    
'''
img: the image to be segmented
algorithm: algorithm to use for segmentation. Defalut: naive way, 8 side nbr

Segmentation object has
segment_list: entire list of segments
get_diagram(algorithm='largest'): returns the diagram segment.
get_label_list()

'''
class BinarizedSegmentation:
    def __init__(self, img, bin_alg='otsu', seg_alg='naive', dgm_crit='area', min_area=25):
        bin_img = binarize(img, alg=bin_alg)
        self.bin_img = bin_img
        segment_list = []
        self.dgm_seg = None
        
        if seg_alg == 'naive':
            kernel = np.ones((3,3), np.uint8)
            labeled, nr_objects = ndimage.label(bin_img, structure=kernel)
            slices = ndimage.find_objects(labeled)
            for idx, s in enumerate(slices):
                cond = labeled[s]==(idx+1)
                seg_img = bool_img(img[s], cond)
                loc = (s[1].start,s[0].start)
                segment = ImageSegment(seg_img, loc, bin_img[s]*cond,img[s])
                if segment.area > min_area:
                    segment_list.append(segment)
                
        if dgm_crit == 'area':
            largest_idx = -1
            largest_area = 0
            for idx, segment in enumerate(segment_list):
                if segment.area > largest_area:
                    largest_idx = idx
                    largest_area = segment.area
            self.dgm_seg = segment_list[largest_idx]
            segment_list[largest_idx] = segment_list[0]
            segment_list[0] = self.dgm_seg
            self.label_seg_list = segment_list[1:]
            self.segment_list = segment_list
    
    def save(self, folderpath, digit=2):
        filepath = next_name(folderpath, digit, 'png')
        cv2.imwrite(filepath, self.dgm_seg.bin_img)
        for seg in self.label_seg_list:
            filepath = next_name(folderpath, digit, 'png')
            cv2.imwrite(filepath, seg.slice_img)
