'''
Created on Apr 14, 2014

@author: minjoon
'''
import numpy as np
import cv2
import csv
import os

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
        if char == "":
            os.remove(self.path+name)
        else:
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
