'''
Created on May 12, 2014

@author: minjoon
'''
from geosol_v1.external.path import find_file
from geosol_v1.geometry.diagram_graph import DiagramGraph
from geosol_v1.geometry.vp_selector import VPSelector
from geosol_v1.image.low_level import open_img, BinarizedSegmentation
from geosol_v1.image.visual_primitive import VPGenerator
import os

import cv2


# returns bgr_img, dg
def init_graph(problempath, bin_seg):
    imgpath = find_file(problempath, 'original')
    bgr_img = open_img(imgpath, color='BGR')
    img = cv2.cvtColor(bgr_img, cv2.cv.CV_BGR2GRAY)

    bin_seg.save(problempath)
    vpg = VPGenerator(bin_seg)
    vpg.save(os.path.join(problempath,'preopt.png'),img)
    vps = VPSelector(bin_seg, vpg)
    vps.save(os.path.join(problempath,'postopt.png'),img)
    dg = DiagramGraph(vps)
    return (bgr_img, dg)

def init_seg(problempath):
    imgpath = find_file(problempath, 'original')
    bgr_img = open_img(imgpath, color='BGR')
    img = cv2.cvtColor(bgr_img, cv2.cv.CV_BGR2GRAY)

    bin_seg = BinarizedSegmentation(img)
    return bin_seg

def ocr_save(img, filepath):
    cv2.imwrite(filepath, img)
    

# saves queried image to imagepath
def query(bgr_img, dg, query, imagepath):
    bgr_img = bgr_img.copy()
    ge_list = []
    vx_list = []
    if query:
        vx_comb_list, ge_comb_list = dg.query(query,True)
        if len(vx_comb_list):
            ge_list.extend(ge_comb_list[0])
            vx_list.extend(vx_comb_list[0])
    dg.draw(bgr_img,ge_list=ge_list,vx_list=vx_list)
    cv2.imwrite(imagepath, bgr_img)