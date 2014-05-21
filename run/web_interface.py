'''
Created on May 12, 2014

@author: minjoon
'''
from geosol_v1.external.path import find_file
from geosol_v1.geometry.diagram_graph import DiagramGraph
from geosol_v1.geometry.vp_selector import VPSelector
from geosol_v1.image.low_level import open_img, BinarizedSegmentation
from geosol_v1.image.visual_primitive import VPGenerator
from geosol_v1.nlp.temp import Word
import os

import cv2


# returns bgr_img, dg
def init_graph(problempath, bin_seg):
    imgpath = find_file(problempath, 'original')
    bgr_img = open_img(imgpath, color='BGR')
    img = cv2.cvtColor(bgr_img, cv2.cv.CV_BGR2GRAY)

    bin_seg.save(problempath)
    vpg = VPGenerator(bin_seg)
    vpg.save(os.path.join(problempath,'preopt.png'),img, ratio=1.5)
    vps = VPSelector(bin_seg, vpg)
    vps.save(os.path.join(problempath,'postopt.png'),img)
    dg = DiagramGraph(vps)
    dg.assign_labels(bin_seg.label_seg_list)
    for vx in dg.vx_list:
        print vx.label
    return (bgr_img, dg)

def init_seg(problempath):
    imgpath = find_file(problempath, 'original')
    bgr_img = open_img(imgpath, color='BGR')
    img = cv2.cvtColor(bgr_img, cv2.cv.CV_BGR2GRAY)

    bin_seg = BinarizedSegmentation(img)
    return bin_seg

def ocr_save(img, filepath):
    cv2.imwrite(filepath, img)
    

def list_ve(dg):
    string = ""
    # enumerate lines
    vc_list, ec_list = dg.query('l',False)
    for combo in vc_list:
        string += "line %s%s," %(combo[0].label,combo[1].label)
    
    # enumerate arcs
    vc_list, ec_list = dg.query('a',False)
    for combo in vc_list:
        string += "arc %s%s," %(combo[0].label,combo[1].label)
       
    # circle 
    for vx in dg.vx_list:
        if vx.idx in dg.arc_graph:
            string += "circle %s," %vx.label
            
    # triangle
    vc_list, ec_list = dg.query('lll',True)
    for combo in vc_list:
        string += "triangle %s%s%s," %(combo[0].label,combo[1].label,combo[2].label)

            
    # angle
    vc_list, ec_list = dg.query('ll',False)
    for combo in vc_list:
        string += "angle %s%s%s," %(combo[0].label,combo[1].label,combo[2].label)
    
    # points
    for vx in dg.vx_list:
        string += "point %s," %vx.label
    
    return string

# saves queried image to imagepath
def query(bgr_img, dg, query, imagepath, ratio=1.5):
    bgr_img = cv2.resize(bgr_img, (0,0), fx=ratio, fy=ratio)
    ge_list = []
    vx_list = []
    if query:
        
        if query == "all":
            dg.draw(bgr_img,ratio=ratio)
            cv2.imwrite(imagepath, bgr_img)
            return
        
        input_array = query.split(' ')
        if len(input_array) == 1:
            ref = query
            if len(ref) == 1:
                shape = 'point'
            else:
                shape = 'line'
        else:
            shape, ref = input_array
            shape = shape.lower()

        if shape == 'circle':
            vx_list, ge = dg.simple_query(shape, ref)
            ge_list.append(ge)
        elif shape in ['line','chord','tangent','secant','diameter']:
            vx_list, ge = dg.simple_query('line', ref)
            ge_list.append(ge)
        elif shape == 'arc':
            vx_list, ge_list = dg.simple_query(shape,ref)
        elif shape == 'point':
            vx_list, ge_list = dg.simple_query(shape,ref)
        else:
            if shape == 'triangle':
                recall = True
                seq = 'lll'
            elif shape == 'angle':
                recall = False
                seq = 'll'
            elif shape == 'pie':
                recall = True
                seq = 'lal'
            vx_comb_list, ge_comb_list = dg.query(seq,recall,all=True)
            if len(vx_comb_list):
                for idx0, vx_comb in enumerate(vx_comb_list):
                    cond = True
                    for idx1, vx in enumerate(vx_comb):
                        if vx.label != ref[idx1]:
                            cond = False
                            break
                    if cond:
                        ge_list.extend(ge_comb_list[idx0])
                        vx_list.extend(vx_comb_list[idx0])
                        break
                        
    dg.draw(bgr_img,ge_list=ge_list,vx_list=vx_list,ratio=ratio)
    cv2.imwrite(imagepath, bgr_img)