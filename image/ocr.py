'''
Created on Apr 14, 2014

@author: minjoon
'''
import numpy as np
import cv2
import csv
import os
from geosol_v1.external.path import next_name

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
cost function between features
'''
def cost_fn(feature0, feature1):
    feature0 = np.array(feature0)
    feature1 = np.array(feature1)
    return np.linalg.norm(feature0-feature1)
    
class LabelRecognizer:
    def __init__(self, path='data/ocr/', digit=6, data_filename='data.csv', img_len=16):
        self.path =path
        self.digit = digit
        self.feature_list = []
        self.char_list = []
        self.img_len = img_len
        
        # initialize feature dictionary
        filepath = os.path.join(path,data_filename)
        if os.path.exists(filepath):
            data_fh = open(filepath, 'rb')
            reader = csv.reader(data_fh, delimiter=',')
            for row in reader:
                self.char_list.append(row[1])
                self.feature_list.append([int(e) for e in row[2:]])
            data_fh.close()
        self.data_fh = open(filepath, 'a')
    
    '''
    save img to db and update feature list as well
    '''
    def save(self, img, char):
        name = next_name(self.path, self.digit, extension='png')
        cv2.imwrite(os.path.join(self.path,name), img)
        new_img = label_resize(img, target_len=self.img_len)
        flat_img = [int(e) for e in new_img.flatten()]
        self.feature_list.append(flat_img)
        self.char_list.append(char)
        imgid = ','.join([str(e) for e in flat_img])
        string = "%s,%s,%s\n" %(name,char,imgid)
        self.data_fh.write(string)
    
    '''
    returns (matched_char,cost) pair
    ''' 
    def recognize(self, img, max_cost=100):
        if len(self.feature_list) > 0:
            new_img = label_resize(img, target_len=self.img_len) 
            flat_img = new_img.flatten()
            cost_list = [cost_fn(flat_img,feature) for feature in self.feature_list]
            idx = np.argmin(cost_list)
            return (self.char_list[idx], cost_list[idx])
        else:
            return ("", 9999)
    
    def batch_save(self, imgs, chars):
        for idx, img in enumerate(imgs):
            char = chars[idx]
            self.save(img,char)
            
    def close(self):
        self.data_fh.close()
