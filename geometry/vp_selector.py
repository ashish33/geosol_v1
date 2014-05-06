'''
Selecting best visual primitives

Created on Apr 29, 2014

@author: minjoon
'''
from geometry.util import line2pt_dist, cir2pt_dist
from image.visual_primitive import VPLine, VPArc, VPGenerator
import numpy as np
import random


# optimization element 
class OptimizationElement:
    def __init__(self, vp, pxl_set, eps=3):
        self.vp = vp
        self.pxl_set = set()
        if type(vp) == VPLine:
            self.pxl_set = \
            set([pxl for pxl in pxl_set if line2pt_dist(vp.line_tuple,pxl) < eps])
        elif type(vp) == VPArc:
            self.pxl_set = \
            set([pxl for pxl in pxl_set if cir2pt_dist(vp.arc_tuple,pxl) < eps])
            
class OptimizationSet:
    def __init__(self):
        self.element_list = []
        self.pxl_set = set()
        self.redun_set = set()
        
    def add(self, element):
        self.element_list.append(element)
        self.redun_set = self.redun_set.union(self.pxl_set.intersection(element.pxl_set))
        self.pxl_set = self.pxl_set.union(element.pxl_set)
        
    def copy(self):
        new_set = OptimizationSet()
        new_set.element_list = self.element_list[:]
        new_set.pxl_set = self.pxl_set.copy()
        new_set.redun_set = self.redun_set.copy()
        return new_set
    
    def __repr__(self):
        return 'P=%d, R=%d' %(len(self.pxl_set),len(self.redun_set))

def obj_func(opt_set, params):
    p,s = params
    P = len(opt_set.pxl_set)
    S = -len(opt_set.redun_set)
    return p*P + s*S

def get_max_val(func, opt_set, params, element_list):
    max_val = -np.inf 
    max_idx = -1
    for idx, element in enumerate(element_list):
        new_opt_set = opt_set.copy()
        new_opt_set.add(element)
        curr_val = func(new_opt_set, params) 
        if curr_val > max_val:
            max_val = curr_val
            max_idx = idx
    return (max_val, max_idx)
        
class VPSelector(VPGenerator):
    def __init__(self, bin_seg, vp_gen, params=None, factor=3):
        self.vpline_list = []
        self.vparc_list = []
        self.line_params = [] 
        self.circle_params = [] 
        
        opt_set = OptimizationSet()
        choice_list = []
        if params == None:
            params = (1,0.5)

        # randomly choose 500 points
        pxl_list = [tuple(elem) for elem in bin_seg.dgm_seg.nz_pts]
        pxl_set = set(pxl_list[0::factor])
        
        for vp in vp_gen.get_vp_list():
            opt_element = OptimizationElement(vp, pxl_set)
            choice_list.append(opt_element)
        
        prev_val = 0    
        val, idx = get_max_val(obj_func, opt_set, params, choice_list)
        while val-prev_val > 0:
            opt_set.add(choice_list.pop(idx))
            print opt_set 
            prev_val = val
            val, idx = get_max_val(obj_func, opt_set, params, choice_list)
            
        vp_list = [element.vp for element in opt_set.element_list]
        for vp in vp_list:
            if type(vp) == VPLine:
                self.vpline_list.append(vp)
            elif type(vp) == VPArc:
                self.vparc_list.append(vp)
        