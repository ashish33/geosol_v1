'''
Created on Apr 29, 2014

@author: minjoon
'''
from abc import ABCMeta, abstractmethod
from geosol_v1.geometry.util import segment_line, segment_arc, line2line_ix, \
    line2arc_ix, non_maximum_suppression, line2pt_dist, parallel_dist, pt2pt_dist, \
    arc2pt_dist, pt2pt_angle
from geosol_v1.image.util import draw_line, draw_arc

import numpy as np


class GraphEdge:
    __metaclass__ = ABCMeta
    @abstractmethod
    def __repr__(self):
        pass
    
    
class GELine(GraphEdge):
    def __init__(self, vp_line, start=0, end=1):
        self.parent = vp_line
        self.abs_line_tuple = segment_line(vp_line.abs_line_tuple, start, end)
        
    def __repr__(self):
        out = "l,%.1f,%.1f,%.1f,%.1f," %tuple(self.abs_line_tuple)
        return out
        
class GEArc(GraphEdge):
    def __init__(self, vp_arc, start_angle=0, end_angle=0):
        self.parent = vp_arc
        self.abs_arc_tuple = np.append(vp_arc.abs_arc_tuple[:3],[start_angle,end_angle])
        
    def __repr__(self):
        out = 'a,%.1f,%.1f,%.1f,%.1f,%.1f,' %tuple(self.abs_arc_tuple)
        return out
        
class Vertex:
    def __init__(self, loc):
        self.loc = tuple(loc) # (x,y) tuple
        self.vx_dict = {} # vx : 'd': direct neighbor, 'i': indirect neighbor
        
    def add_vx(self, vx, key):
        self.vx_dict[vx] = key
            
    def __hash__(self):
        return hash(self.loc)
    
    def __repr__(self):
        return "(%.1f,%.1f,)" %self.loc
        
        
class DiagramGraph:
    def __init__(self, vps, eps=10):
        self.vx_list = []
        self.geline_list = []
        self.gearc_list = []
        # these are for reference
        self.vpline_list = vps.vpline_list
        self.vparc_list = vps.vparc_list
        
        # construct vertices via intersection
        temp_vxpt_list = []
        # line2line ix
        for idx0, vpline0 in enumerate(self.vpline_list):
            for vpline1 in self.vpline_list[idx0+1:]:
                line0 = vpline0.abs_line_tuple
                line1 = vpline1.abs_line_tuple
                pt = line2line_ix(line0,line1)
                if pt != None:
                    temp_vxpt_list.append(pt)
                                
        #line2arc ix
        for vpline in self.vpline_list:
            for vparc in self.vparc_list:
                line = vpline.abs_line_tuple
                arc = vparc.abs_arc_tuple
                temp_vxpt_list.extend(line2arc_ix(line,arc))

        # Add end points
        for vpline in self.vpline_list:
            temp_vxpt_list.append(vpline.abs_line_tuple[:2])
            temp_vxpt_list.append(vpline.abs_line_tuple[2:])
       
        # Add centers 
        for vparc in self.vparc_list:
            temp_vxpt_list.append(vparc.abs_arc_tuple[:2])
        
        # NMS on the vertices
        [self.vx_list.append(Vertex(e)) for e in non_maximum_suppression(temp_vxpt_list,eps)]
        
        # construct graph edges
        self.geline_list = []
        self.gearc_list = []
        self.line_graph = {} # (vx0,vx1) : GELine
        self.arc_graph = {} # (vx0,vx1) : GEArc
        
        # construct GELine first
        for vpline in self.vpline_list:
            line = vpline.abs_line_tuple
            # obtain all relevant vertices (those that the line passes through)
            vx_list = [vx for vx in self.vx_list if line2pt_dist(line,vx.loc) < eps]
            # order the vxpt in one direction
            linelen = pt2pt_dist(line[:2],line[2:])
            # normalized parallel distance
            norm_pardist_list = [0.5+parallel_dist(line,vx.loc)/linelen for vx in vx_list]
            idx_list = np.argsort(norm_pardist_list)
             
            # for each consecutive pair, create edges
            for i, idx in enumerate(idx_list[:-1]):
                next_idx = idx_list[i+1]
                start = norm_pardist_list[idx]
                end = norm_pardist_list[next_idx]
                geline = GELine(vpline, start, end)
                self.geline_list.append(geline)
                # Add the edge to the graph
                self.line_graph[(vx_list[idx],vx_list[next_idx])] = geline
                # Add direct neighbor
                vx_list[idx].add_vx(vx_list[next_idx],'d')
                vx_list[next_idx].add_vx(vx_list[idx],'d')
                
            # Add indirect neighbor
            for idx, vx0 in enumerate(vx_list):
                for vx1 in vx_list[idx+1:]:
                    if vx1 not in vx0.vx_dict:
                        vx0.add_vx(vx1, 'i')
                    if vx0 not in vx1.vx_dict:
                        vx1.add_vx(vx0, 'i')
            
        # construct GEArc
        for vparc in self.vparc_list:
            arc = vparc.abs_arc_tuple
            # obtain all relevant vertices
            vx_list = [vx for vx in self.vx_list if arc2pt_dist(arc, vx.loc) < eps]
            angle_list = [pt2pt_angle(arc[:2],vx.loc) for vx in vx_list]
            idx_list = np.argsort(angle_list)
            
            for i, idx in enumerate(idx_list):
                # for now, assume only circle. Later implement Arc
                if i == len(idx_list)-1:
                    next_idx = idx_list[0]
                else:
                    next_idx = idx_list[i+1]
                start = angle_list[idx]
                end = angle_list[next_idx]
                if start > end:
                    start -= 2*np.pi
                gearc = GEArc(vparc, start, end)
                self.gearc_list.append(gearc)
                key = (vx_list[idx],vx_list[next_idx])
                # it is possible to have more than one arc between two points
                if key in self.arc_graph:
                    self.arc_graph[key].append(gearc)
                else:
                    self.arc_graph[key] = [gearc]
                vx_list[idx].add_vx(vx_list[next_idx],'d')
                vx_list[next_idx].add_vx(vx_list[idx],'d')
                
            for idx, vx0 in enumerate(vx_list):
                for vx1 in vx_list[idx+1:]:
                    if vx1 not in vx0.vx_dict:
                        vx0.add_vx(vx1, 'i')
                    if vx0 not in vx1.vx_dict:
                        vx1.add_vx(vx0, 'i')
    
    '''
    seq="lll", recall=True will represent a triangle
    seq="ll", recall=False will represent an angle
    seq="lal", recall=True will represent a pie
    revisit=True will allow visiting already visited vertices
    '''
    def query(self, seq, recall, revisit=False):
        comb_list = []
        ge_comb_list = []
        def helper(vx_list, ge_list):
            # base case
            if recall and len(vx_list) == len(seq):
                ge = ge_list_has(self.get_ge_list(vx_list[0], vx_list[-1]), seq[-1])
                if ge:
                    comb_list.append(vx_list)
                    ge_list.append(ge)
                    ge_comb_list.append(ge_list)
            elif not recall and len(vx_list) > len(seq):
                comb_list.append(vx_list)
                ge_comb_list.append(ge_list)
            else:
                idx = len(vx_list) - 1
                for vx in vx_list[idx].vx_dict:
                    if vx not in vx_list or revisit:
                        ge = ge_list_has(self.get_ge_list(vx_list[idx],vx),seq[idx])
                        if ge:
                            new_vx_list = vx_list[:]
                            new_vx_list.append(vx)
                            new_ge_list = ge_list[:]
                            new_ge_list.append(ge)
                            helper(new_vx_list, new_ge_list)
        for vx in self.vx_list:
            helper([vx],[])
        return (comb_list, ge_comb_list)
    
    def get_ge_list(self, vx0, vx1):
        ge_list = []
        key0 = (vx0,vx1)
        key1 = (vx1,vx0)
        if key0 in self.line_graph:
            ge_list.append(self.line_graph[key0])
        elif key1 in self.line_graph:
            ge_list.append(self.line_graph[key1])
        
        if key0 in self.arc_graph:
            ge_list.extend(self.arc_graph[key0])
        elif key1 in self.arc_graph:
            ge_list.extend(self.arc_graph[key1])
            
        return ge_list
    
            
    '''
    Draw lines, arcs, and vertices on img (destructive)
    '''
    def draw(self, bgr_img, ge_list=None, vx_list=None):
        edge_color = (255,0,0)
        edge_width = 2
        vertex_color = (0,0,255)
        vertex_width = 1
        vertex_radius = 4
        
        if ge_list == None:
            ge_list = []
            ge_list.extend(self.geline_list)
            ge_list.extend(self.gearc_list)
        elif isinstance(ge_list, GELine) or isinstance(ge_list, GEArc):
            ge_list = [ge_list]
        if vx_list == None:
            vx_list = self.vx_list
            
        # draw edges
        for ge in ge_list:
            if isinstance(ge,GELine):
                print "drawing %s" %ge
                draw_line(bgr_img, ge.abs_line_tuple, edge_color, edge_width)
            elif isinstance(ge,GEArc):
                print "drawing %s" %ge
                draw_arc(bgr_img, ge.abs_arc_tuple, edge_color, edge_width)
        
        # draw vertices
        for vx in vx_list:
            arc_tuple = np.append(vx.loc, [vertex_radius,0,0])
            draw_arc(bgr_img, arc_tuple, vertex_color, vertex_width)

def ge_list_has(ge_list, string):
    for ge in ge_list:
        if (isinstance(ge,GELine) and string=='l') or \
        (isinstance(ge,GEArc) and string=='a'):
            return ge
    return None
            

        