'''
Created on Apr 14, 2014

@author: minjoon
'''
from abc import ABCMeta, abstractmethod
import cv2
import numpy as np
import matplotlib.pyplot as plt

class VisualPrimitive:
    __metaclass__ = ABCMeta
    @abstractmethod
    def __repr__(self):
        pass
    
class VPLine(VisualPrimitive):
    def __init__(self, line_tuple):
        self.line_tuple = line_tuple
        self.label_list = [] # first two elements are starting points
        
    def __repr__(self):
        return 'l,%.1f,%.1f,%.1f,%.1f' %self.line_tuple

class VPArc(VisualPrimitive):
    '''
    arc_tuple: (x,y,r,t0,t1)
    '''
    def __init__(self, arc_tuple):
        self.arc_tuple = arc_tuple
        self.label_list = [] # first element is the center
        
    def __repr__(self):
        return 'a,%.1f,%.1f,%.1f,%.1f,%.1f' %self.arc_tuple
    
'''
Given an image,
generate a list of visual primitives

line_params:  (rho, theta, line_mg, line_ml, th, nms_rho, nms_theta)

constants: rho = 1, theta = pi/180, max_gap = 3
variables: th, nms_rho, nms_theta

circle_params: (dp, minRadius, maxRadius, arc_mg, arc_ml, params1, params2, minDist)

constants: dp = 1, minRadius = 10, maxRadius = 200, max_gap = 3, params1 = 
variables: params2 (th), minDist (for nms)

'''
class VPGenerator:
    def __init__ (self, segment, line_params=None, circle_params=None, eps=1.5):
        nz_pts = segment.nz_pts

        if line_params == None:
            line_params = (1,np.pi/180,3,20,30,2,np.pi/60)
        if circle_params == None:
            circle_params = (1,0,0,3,20,30,35,2)
        rho, theta, line_mg, line_ml, th, nms_rho, nms_theta = line_params
        dp, minRadius, maxRadius, arc_mg, arc_ml, param1, param2, minDist = circle_params
        method = cv2.cv.CV_HOUGH_GRADIENT
        rt_list = cv2.HoughLines(segment.bin_img,rho,theta,th)[0]
        circle_list = cv2.HoughCircles(segment.img,method,dp,minDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)[0]
        
        # non-maximal suppression
        nms_rt_list = rt_nms(rt_list, nms_rho, nms_theta)

        self.vpline_list = []
        self.vparc_list = []
        for r,t in nms_rt_list:
            line_tuple_list = rt2lines(nz_pts,r,t,line_mg,line_ml,eps)
            for line_tuple in line_tuple_list:
                self.vpline_list.append(VPLine(line_tuple))

        ''' 
        for x,y,r in circle_list:
            arc_tuple_list = circle2arcs(nz_pts,x,y,r,arc_mg,arc_ml,eps)
            for arc_tuple in arc_tuple_list:
                self.vparc_list.append(VPArc(arc_tuple))
        '''
                
    def get_vp_list(self):
        temp_list = self.vpline_list[:]
        temp_list.extend(self.vparc_list)
        return temp_list
   
    # visual primitives with translation based onthe location of the segment 
    def get_abs_vp_list(self):
        pass

# non-maximal suppression for rho-theta list
def rt_nms(rt_list, nms_rho, nms_theta):
    out_list = [rt_list[0]]
    for r,t in rt_list[1:]:
        cond = True
        for rr,tt in out_list:
            if np.abs(r-rr)<nms_rho and np.abs(t-tt)<nms_theta:
                cond = False
                break
        if cond:
            out_list.append((r,t))
                
    return np.array(out_list)

def rt_dist(r, t, x, y):
    return np.abs(r-(x*np.cos(t)+y*np.sin(t)))


def rt2pts(nz_pts, r, t, eps=1.5):
    pt_list = []
    for pt in nz_pts:
        if rt_dist(r,t,pt[0],pt[1]) < eps:
            pt_list.append(pt)
    return np.array(pt_list)

def rt2lines(nz_pts, r, t,line_mg, line_ml, eps=1.5):
    pts = rt2pts(nz_pts, r, t, eps=eps)
    u = np.array([np.sin(t), -np.cos(t)])
    pt0 = pts[0]
    dist_list = [np.dot(pt-pt0,u) for pt in pts]
    order = np.argsort(dist_list)
    lines = []
    start_idx = None
    end_idx = None
    for order_idx, idx in enumerate(order):
        if start_idx == None:
            start_idx = idx
            end_idx = idx
        elif np.abs(dist_list[idx]-dist_list[order[order_idx-1]]) > line_mg or order_idx == len(order)-1:
            # add only if the length of line > minLen
            if np.abs(dist_list[start_idx]-dist_list[end_idx]) > line_ml:
                lines.append(np.append(pts[start_idx],pts[end_idx]))
            start_idx = None
        else:
            end_idx = idx
    return lines


def circle2arcs(img, x, y, r, arc_mg, arc_ml, eps=1.5):
    return []
                
            
        
        
