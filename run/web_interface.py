'''
Created on May 12, 2014

@author: minjoon
'''
from geosol_v1.external.path import find_file
from geosol_v1.geometry.diagram_graph import DiagramGraph
from geosol_v1.geometry.util import pt2pt_dist, line2line_angle
from geosol_v1.geometry.vp_selector import VPSelector
from geosol_v1.image.low_level import open_img, BinarizedSegmentation
from geosol_v1.image.visual_primitive import VPGenerator
from geosol_v1.nlp.temp import Word
import os

import numpy as np
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
    
    
class VE:
    def __init__(self, ge_list, vx_list, shape):
        self.ge_list = ge_list
        self.vx_list = vx_list
        self.shape = shape
        self.ppt = {}
        if shape == 'line':
            line_tuple = self.ge_list[0].abs_line_tuple
            self.ppt['length'] = pt2pt_dist(line_tuple[:2],line_tuple[2:])
        elif shape == 'arc':
            arc_tuple = self.ge_list[0].abs_arc_tuple
            self.ppt['length'] = np.abs(arc_tuple[3]-arc_tuple[4])*arc_tuple[2]
        elif shape == 'circle':
            arc_tuple = self.ge_list[0].abs_arc_tuple
            self.ppt['radius'] = arc_tuple[2]
        elif shape == 'angle':
            line0 = np.append(vx_list[0].loc,vx_list[1].loc)
            line1 = np.append(vx_list[1].loc,vx_list[2].loc)
            self.ppt['angle'] = line2line_angle(line0,line1)
        elif shape == 'triangle':
            line0 = np.append(vx_list[0].loc,vx_list[1].loc)
            line1 = np.append(vx_list[1].loc,vx_list[2].loc)
            line2 = np.append(vx_list[2].loc,vx_list[0].loc)
            lines = [line0,line1,line2]
            angles = [line2line_angle(lines[0],lines[1]), line2line_angle(lines[1],lines[2]), line2line_angle(lines[2],lines[0])]
            right_errors = [error(angle,np.pi/2) for angle in angles]
            self.ppt['right'] = (1-min(right_errors))*100

            diffs = [error(angles[0],angles[1]),error(angles[1],angles[2]),error(angles[2],angles[0])]
            self.ppt['iso'] = (1-min(diffs))*100
                
            
            
        
    def __repr__(self):
        string = self.shape + " "
        for v in self.vx_list:
            string += v.label
        return normalize_query(string)
        
def list_rel(dg, query):
    if query == "all":
        return []
    ve_list = {}
    query = normalize_query(query)
    
    #insert lines
    vc_list, ec_list = dg.query('l',False)
    for idx, vc in enumerate(vc_list):
        ec = ec_list[idx]
        ve = VE(ec,vc,'line')
        ve_list[repr(ve)] = ve
        
    # enumerate arcs
    vc_list, ec_list = dg.query('a',False)
    for idx, vc in enumerate(vc_list):
        ec = ec_list[idx]
        ve = VE(ec,vc,'arc')
        ve_list[repr(ve)] = ve
        
    # circle
    for vx in dg.vx_list:
        if vx.idx in dg.arc_graph:
            ve = VE([dg.arc_graph[vx.idx][0]],[vx],'circle')
            ve_list[repr(ve)] = ve
            
    # triangle
    vc_list, ec_list = dg.query('lll',True)
    for idx, vc in enumerate(vc_list):
        ec = ec_list[idx]
        ve = VE(ec,vc,'triangle')
        ve_list[repr(ve)] = ve
    
    # angle
    vc_list, ec_list = dg.query('ll',False)
    for idx, vc in enumerate(vc_list):
        ec = ec_list[idx]
        ve = VE(ec,vc,'angle')
        ve_list[repr(ve)] = ve
        
    # points
    for vx in dg.vx_list:
        ve = VE([],[vx],'point')
        ve_list[repr(ve)] = ve
        
    attr_list = []
     
    query_ve = ve_list[query]
    if query_ve.shape == 'line':
        attr_list.append("Length of %s is %1.f pixels." %(query,query_ve.ppt['length']))
    elif query_ve.shape == 'arc':
        attr_list.append("Length of %s is %1.f pixels." %(query,query_ve.ppt['length']))
    elif query_ve.shape == 'circle':
        attr_list.append("The radius of %s is %1.f pixels." %(query,query_ve.ppt['radius']))
    elif query_ve.shape == 'angle':
        attr_list.append("%s = %1.f degrees." %(query,query_ve.ppt['angle']*180/np.pi))
    elif query_ve.shape == 'triangle':
        if query_ve.ppt['right'] > 90:
            attr_list.append("%s is a right triangle(%1.f%%)." %(query,query_ve.ppt['right']))
        if query_ve.ppt['iso'] > 90:
            attr_list.append("%s is an isosceles triangle (%1.f%%)." %(query,query_ve.ppt['iso']))
            
    for ve in ve_list.values():
        if ve == query_ve:
            continue
        if query_ve.shape in ['line','arc'] and ve.shape in ['line','arc']:
            len_err = error(query_ve.ppt['length'],ve.ppt['length'])
            if len_err < 0.1: 
                conf = (1-len_err)*100
                attr_list.append("%s and %s are equal in length (%1.f%%)." %(query_ve,ve,conf))
           
            ''' 
            angle = line2line_angle(query_ve.ge_list[0].abs_line_tuple,ve.ge_list[0].abs_line_tuple)
            ang_err = error(angle,np.pi/2)
            if ang_err < 0.1:
                conf = (1-ang_err)*100
                attr_list.append("%s is perpendicular to %s (%1.f%%)." %(query_ve,ve,conf))
            '''
        
        if query_ve.shape == 'line' and ve.shape == 'circle':
            if ve.vx_list[0] in query_ve.vx_list:
                len_err = error(query_ve.ppt['length'],ve.ppt['radius'])
                if len_err < 0.1:
                    conf = (1-len_err)*100
                    attr_list.append("%s is a radius of %s (%1.f%%)." %(query_ve,ve,conf))
                    
        if query_ve.shape == 'circle' and ve.shape == 'line':
            if query_ve.vx_list[0] in ve.vx_list:
                len_err = error(ve.ppt['length'],query_ve.ppt['radius'])
                if len_err < 0.1:
                    conf = (1-len_err)*100
                    attr_list.append("%s is a radius of %s (%1.f%%)." %(ve,query_ve,conf))
        
        if query_ve.shape == 'angle' and ve.shape == 'angle':
            ang_err = error(query_ve.ppt['angle'],ve.ppt['angle'])
            if ang_err < 0.1:
                conf = (1-ang_err)*100
                attr_list.append("%s = %s (%1.f%%)." %(query_ve,ve,conf))
                
        if query_ve.shape == 'point' and ve.shape == 'circle':
            if query_ve.vx_list[0] == ve.vx_list[0]:
                attr_list.append("%s is the center of %s." %(query_ve,ve))
                
    return attr_list
        

        
    
def error(val1, val2):
    return 2.0*np.abs(val1-val2)/(val1+val2)
    

def list_ve(dg):
    arr = []
    # enumerate lines
    vc_list, ec_list = dg.query('l',False)
    for combo in vc_list:
        arr.append("line %s%s" %(combo[0].label,combo[1].label))
    
    # enumerate arcs
    vc_list, ec_list = dg.query('a',False)
    for combo in vc_list:
        arr.append("arc %s%s" %(combo[0].label,combo[1].label))
       
    # circle 
    for vx in dg.vx_list:
        if vx.idx in dg.arc_graph:
            arr.append("circle %s" %vx.label)
            
    # triangle
    vc_list, ec_list = dg.query('lll',True)
    for combo in vc_list:
        arr.append("triangle %s%s%s" %(combo[0].label,combo[1].label,combo[2].label))

            
    # angle
    vc_list, ec_list = dg.query('ll',False)
    for combo in vc_list:
        arr.append("angle %s%s%s" %(combo[0].label,combo[1].label,combo[2].label))
    
    # points
    for vx in dg.vx_list:
        arr.append("point %s" %vx.label)
        
    for idx, e in enumerate(arr):
        arr[idx] = normalize_query(e)
    
    return ','.join(arr)

def normalize_query(query):
    arr = query.split(' ')
    if len(arr) == 1:
        if len(query) == 1:
            return "point " + query
        else:
            return "line " + order_ref(query,True)
    elif len(arr) == 2:
        if arr[0].lower() in ['line','chord','tangent','secant','diameter']:
            return 'line ' + order_ref(arr[1],True)
        if arr[0] == 'triangle':
            return arr[0] + " " + order_ref(arr[1],False)
        else:
            return arr[0] + " " + order_ref(arr[1],True)

# if seq == True, then seq matters
def order_ref(ref, seq=False):
    if seq:
        return min(ref,ref[::-1])
    else:
        arr = [x for x in ref]
        arr.sort()
        return "".join(arr)
            

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