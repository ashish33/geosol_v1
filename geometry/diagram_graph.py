'''
Created on Apr 29, 2014

@author: minjoon
'''
from abc import ABCMeta, abstractmethod
import os

import cv2
import numpy as np

from geometry.util import segment_line
from image.util import draw_line, draw_arc


class GraphEdge:
    __metaclass__ = ABCMeta
    @abstractmethod
    def __repr__(self):
        pass
    
    
class GELine(GraphEdge):
    def __init__(self, vp_line, start=0, end=1):
        self.parent = vp_line
        self.abs_line_tuple = segment_line(vp_line.abs_line_tuple, start, end)
        
class GEArc(GraphEdge):
    def __init__(self, vp_arc, start=0, end=1):
        self.parent = vp_arc
        self.abs_arc_tuple = segment_arc(vp_arc.abs_arc_tuple, start, end)
        
class DiagramGraph:
    def __init__(self, vps):
        # temp
        self.vpline_list = vps.vpline_list
        self.vparc_list = vps.vparc_list
        pass
    
    '''
    Draw lines, arcs, and vertices on img and save it
    '''
    def save(self, folderpath, img):
        # save an image
        imgpath = os.path.join(folderpath, 'graph.png')
        bgr_img = cv2.cvtColor(img,cv2.cv.CV_GRAY2BGR)
        
        edge_color = (255,0,0) # red
        edge_width = 2
        
        # draw lines
        for vpline in self.vpline_list:
            draw_line(bgr_img, vpline.abs_line_tuple, edge_color, edge_width)
        for vparc in self.vparc_list:
            draw_line(bgr_img, vparc.abs_arc_tuple, edge_color, edge_width)
        
        '''
        for geline in self.geline_list:
            draw_line(bgr_img, geline.abs_line_tuple, edge_color, edge_width)
        for gearc in self.gearc_list:
            draw_arc(bgr_img, gearc.abs_arc_tuple, edge_color, edge_width)
        '''
        # draw vertices
        vertex_color = (0,0,255) # blue
        vertex_width = 1
        vertex_radius = 4
        
        for gevertex in self.gevertex_list:
            arc_tuple = np.append(gevertex.loc, vertex_radius)
            draw_arc(bgr_img, gevertex.loc, arc_tuple, vertex_color, vertex_width)
        
        cv2.imwrite(imgpath, bgr_img)
        print 'Successfully saved graph image to %s' %imgpath
        