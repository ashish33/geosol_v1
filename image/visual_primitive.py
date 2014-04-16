'''
Created on Apr 14, 2014

@author: minjoon
'''
from abc import ABCMeta, abstractmethod
from matplotlib import cm
import os

import cv2

from geometry.util import sim_line, sim_circle
from image.low_level import open_img
from image.ocr import LabelRecognizer
import matplotlib.pyplot as plt
import numpy as np
import csv


class VisualPrimitive:
    __metaclass__ = ABCMeta
    @abstractmethod
    def __repr__(self):
        pass
    
    @abstractmethod
    def assign_abs(self, offset):
        pass
    
    @abstractmethod
    def assign_rel(self, offset):
        pass
    
    @abstractmethod
    def add_label(self, label):
        pass
   
    'measure similarity between two same type of primitives' 
    @abstractmethod
    def similarity(self, other):
        pass
    
class VPLine(VisualPrimitive):
    def __init__(self, line_tuple):
        self.line_tuple = line_tuple
        self.pt0 = line_tuple[:2]
        self.pt1 = line_tuple[2:]
        self.label_list = [] # first two elements are starting points
        self.abs_line_tuple = list(line_tuple[:])
        pass
        
    def assign_abs(self, offset):
        self.abs_line_tuple[0] += offset[0]
        self.abs_line_tuple[1] += offset[1]
        self.abs_line_tuple[2] += offset[0]
        self.abs_line_tuple[3] += offset[1]
        
    def assign_rel(self, offset):
        self.line_tuple[0] -= offset[0]
        self.line_tuple[1] -= offset[1]
        self.line_tuple[2] -= offset[0]
        self.line_tuple[3] -= offset[1]
        
    def add_label(self, label):
        self.label_list.append(label)
        
    def similarity(self, other):
        return sim_line((self.pt0,self.pt1),(other.pt0,other.pt1))
        
    def __repr__(self):
        out = 'l,%.1f,%.1f,%.1f,%.1f,' %tuple(self.abs_line_tuple)
        out += ','.join(self.label_list)
        return out

class VPArc(VisualPrimitive):
    '''
    arc_tuple: (x,y,r,t0,t1)
    '''
    def __init__(self, arc_tuple):
        self.arc_tuple = arc_tuple
        self.center = arc_tuple[:2]
        self.radius = arc_tuple[2]
        self.label_list = [] # first element is the center
        self.abs_arc_tuple = list(arc_tuple[:])
        
    def assign_abs(self, offset):
        self.abs_arc_tuple[0] += offset[0]
        self.abs_arc_tuple[1] += offset[1]
        
    def assign_rel(self, offset):
        self.arc_tuple[0] -= offset[0]
        self.arc_tuple[1] -= offset[1]

    def add_label(self, label):
        self.label_list.append(label)
        
    def similarity(self, other):
        sim_circle(self.center,self.radius,other.center,other.radius)
        
    def __repr__(self):
        out = 'a,%.1f,%.1f,%.1f,%.1f,%.1f,' %tuple(self.abs_arc_tuple)
        out += ','.join(self.label_list)
        return out
    
    
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
    def __init__ (self, bin_seg=None, filepath=None, line_params=None, circle_params=None, eps=1.5, label_tol=15):
        self.vpline_list = []
        self.vparc_list = []
        
        # obtain VP info from a solution file (in csv)
        if filepath != None:
            fh = open(filepath, 'r')
            reader = csv.reader(fh, delimiter=',')
            for row in reader:
                if row[0] == 'l':
                    vp = VPLine([float(elem) for elem in row[1:5]])
                    for label in row[5:]:
                        vp.add_label(label)
                    self.vpline_list.append(vp)
                else:
                    vp = VPArc([float(elem) for elem in row[1:6]])
                    for label in row[6:]:
                        vp.add_label(label)
                    self.vparc_list.append(vp)
        else:
    
            segment = bin_seg.dgm_seg
            nz_pts = segment.nz_pts
            loc = segment.loc
    
            if line_params == None:
                line_params = (1,np.pi/180,3,20,30,2,np.pi/60)
            if circle_params == None:
                circle_params = (1,20,100,2,20,50,40,2)
            rho, theta, line_mg, line_ml, th, nms_rho, nms_theta = line_params
            dp, minRadius, maxRadius, arc_mg, arc_ml, param1, param2, minDist = circle_params
            method = cv2.cv.CV_HOUGH_GRADIENT
            rt_list = cv2.HoughLines(segment.bin_img,rho,theta,th)[0]
            circle_list = cv2.HoughCircles(segment.img,method,dp,minDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)[0]
            
            # non-maximal suppression
            nms_rt_list = rt_nms(rt_list, nms_rho, nms_theta)
    
            for r,t in nms_rt_list:
                line_tuple_list = rt2lines(nz_pts,r,t,line_mg,line_ml,eps)
                for line_tuple in line_tuple_list:
                    self.vpline_list.append(VPLine(line_tuple))
    
            for x,y,r in circle_list:
                arc_tuple_list = circle2arcs(nz_pts,x,y,r,arc_mg,arc_ml,eps)
                for arc_tuple in arc_tuple_list:
                    self.vparc_list.append(VPArc(arc_tuple))
                    
            # Assign abs tuple for each visual element
            for vp in self.get_vp_list():
                vp.assign_abs(loc)
                
            # Assign labels to each segment
            for vp in self.vpline_list:
                for x,y in [vp.abs_line_tuple[:2],vp.abs_line_tuple[2:]]:
                    dist_list = [distance(x,y,seg.center[0],seg.center[1]) for seg in bin_seg.label_seg_list]
                    vp.add_label(bin_seg.label_seg_list[np.argmin(dist_list)].label)
                
    def get_vp_list(self):
        temp_list = self.vpline_list[:]
        temp_list.extend(self.vparc_list)
        return temp_list
    
    # save absolute values of primitives
    def save(self, filepath):
        fh = open(filepath, 'w')
        for vp in self.get_vp_list():
            fh.write(repr(vp)+'\n')
        fh.close()
        print 'Successfully saved visual primitives to %s' %filepath
        
   

def distance(x0, y0, x1, y1):
    return np.sqrt((x1-x0)**2+(y1-y0)**2)

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

def circle2arcs(nz_pts, x, y, r, arc_mg, arc_ml, eps=1.5):
    return [(x,y,r,0,0)]

def compare_solution():
    pass

class VPRecorder:
    def __init__(self):
        self.count = 0
        self.segs = []
        self.cirs = []
        self.prev = None
        self.prev2 = None
    
    def click_seg(self, event):
        self.count += 1
        self.curr = (event.ydata, event.xdata)
        if self.count % 2 == 0:
            self.segs.append((self.prev,self.curr))
        else:
            self.prev = self.curr
            
    def click_cir(self, event):
        self.count += 1
        self.curr = (event.ydata, event.xdata)
        if self.count % 3 == 1:
            self.prev = self.curr
        elif self.count %3 == 2:
            self.prev2 = self.curr
        else:
            a = np.array(self.prev)
            b = np.array(self.prev2)
            c = np.array(self.curr)
            p = (a+b)/2.0
            q = (b+c)/2.0
            v = np.array([a[1]-b[1],b[0]-a[0]])
            u = np.array([b[1]-c[1],c[0]-b[0]])
            ln = v[0]*(p[1]-q[1])+v[1]*(q[0]-p[0])
            ld = u[1]*v[0]-v[1]*u[0]
            l = float(ln)/ld
            o = q + l*u
            r = np.linalg.norm(o-a)
            self.cirs.append((o,r))
            
    def record(self, filepath):
        img = open_img(filepath)
        dirpath = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        out_fh = open(os.path.join(dirpath,'gt_vp_sln.csv'), 'a')
        #plotting for gt recording
        # lines
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap=cm.Greys_r)
        ax.set_title("lines for " + filename)
        cid = fig.canvas.mpl_connect('button_press_event', self.click_seg)
        plt.show()
        self.count = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap=cm.Greys_r)
        ax.set_title("circles for " + filename)
        cid = fig.canvas.mpl_connect('button_press_event', self.click_cir)
        plt.show()
        
        plt.imshow(img, cmap=cm.Greys_r)
        for seg in self.segs:
            a,b = seg
            ar,ac = a
            br, bc = b
            xs = [a[1],b[1]]
            ys = [a[0],b[0]]
            plt.plot(xs,ys,'r')
            out_fh.write('l,%.1f,%.1f,%.1f,%.1f\n' %(ac,ar,bc,br))
        
        fig = plt.gcf()
        for cir in self.cirs:
            c,r = cir
            cr, cc = c
            cir_handle=plt.Circle((cc,cr),r,color='r',fill=False)
            fig.gca().add_artist(cir_handle)
            out_fh.write('c,%.1f,%.1f,%.1f,0,0\n' %(cc,cr,r))
        plt.show()

