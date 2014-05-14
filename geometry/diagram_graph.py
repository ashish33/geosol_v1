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
    def __init__(self, loc, idx):
        self.loc = tuple(loc) # (x,y) tuple
        self.nbr = {} # vx : 1 = direct neighbor, 2+ = indirect neighbor
        self.idx = idx
        self.label = None
        
    def add_nbr(self, vx, key):
        self.nbr[vx.idx] = key
        
    def assign_label(self, label):
        self.label = label
            
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
        self.label_dict = {} # label -> vx
        
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
        [self.vx_list.append(Vertex(e,idx)) for idx,e in enumerate(non_maximum_suppression(temp_vxpt_list,eps))]
        
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
            
            for i, vx_idx0 in enumerate(idx_list[:-1]):
                start = norm_pardist_list[vx_idx0]
                for j, vx_idx1 in enumerate(idx_list[i+1:]):
                    end = norm_pardist_list[vx_idx1]
                    geline = GELine(vpline, start, end)
                    self.geline_list.append(geline)
                    key = [vx_list[vx_idx0].idx,vx_list[vx_idx1].idx]
                    key.sort()
                    key = tuple(key)
                    self.line_graph[key] = geline
                    vx_list[vx_idx0].add_nbr(vx_list[vx_idx1], np.abs(i-j))
                    vx_list[vx_idx1].add_nbr(vx_list[vx_idx0], np.abs(i-j))
            
        # construct GEArc
        for vparc in self.vparc_list:
            arc = vparc.abs_arc_tuple
            # obtain all relevant vertices
            vx_list = [vx for vx in self.vx_list if arc2pt_dist(arc, vx.loc) < eps]
            if len(vx_list) > 0:
                angle_list = [pt2pt_angle(arc[:2],vx.loc) for vx in vx_list]
                idx_list = np.argsort(angle_list)
                
                for i, vx_idx0 in enumerate(idx_list[:-1]):
                    start = angle_list[vx_idx0]
                    for j, vx_idx1 in enumerate(idx_list[i+1:]):
                        end = angle_list[vx_idx1]
                        if start > end:
                            start -= 2*np.pi
                        gearc = GEArc(vparc, start, end)
                        self.gearc_list.append(gearc)
                        key = [vx_list[vx_idx0].idx,vx_list[vx_idx1].idx]
                        key.sort()
                        key = tuple(key)
                        if key in self.arc_graph:
                            self.arc_graph[key].append(gearc)
                        else:
                            self.arc_graph[key] = [gearc]
                        vx_list[vx_idx0].add_nbr(vx_list[vx_idx1], np.abs(i-j))
                        vx_list[vx_idx1].add_nbr(vx_list[vx_idx0], np.abs(i-j))
                        
            # also add the circle itself
            dist_list = [pt2pt_dist(arc[:2],vx.loc) for vx in self.vx_list]
            min_idx = np.argmin(dist_list)
            if dist_list[min_idx] < eps:
                gearc = GEArc(vparc, 0, 0)
                self.gearc_list.append(gearc)
                key = self.vx_list[min_idx].idx
                if key in self.arc_graph:
                    self.arc_graph[key].append(gearc)
                else:
                    self.arc_graph[key] = [gearc]
                
    
    '''
    seq="lll", recall=True will represent a triangle
    seq="ll", recall=False will represent an angle
    seq="lal", recall=True will represent a pie
    seq="c" returns all circle
    revisit=True will allow visiting already visited vertices
    '''
    def query(self, seq, recall, revisit=False):
        comb_list = []
        ge_comb_list = []
        def helper(vx_list, ge_list):
            # base case
            if recall and len(vx_list) == len(seq):
                ge = self.get_ge(vx_list[0], vx_list[-1], seq[-1])
                if ge:
                    comb_list.append(vx_list)
                    ge_list.append(ge)
                    ge_comb_list.append(ge_list)
            elif not recall and len(vx_list) > len(seq):
                comb_list.append(vx_list)
                ge_comb_list.append(ge_list)
            else:
                idx = len(vx_list) - 1
                last_vx = vx_list[idx]
                for vx_idx in last_vx.nbr:
                    vx = self.vx_list[vx_idx]
                    if vx not in vx_list or revisit:
                        ge = self.get_ge(vx_list[idx],vx,seq[idx])
                        if ge:
                            new_vx_list = vx_list[:]
                            new_vx_list.append(vx)
                            new_ge_list = ge_list[:]
                            new_ge_list.append(ge)
                            helper(new_vx_list, new_ge_list)
        for vx in self.vx_list:
            helper([vx],[])
        return (comb_list, ge_comb_list)
    
    def simple_query(self, shape, ref):
        if shape == 'circle':
            if ref in self.label_dict:
                vx = self.label_dict[ref]
            else:
                print "cannot find label %s" %ref
                return None
            return ([vx],self.arc_graph[vx.idx][0])
        else:
            if ref[0] in self.label_dict and ref[1] in self.label_dict:
                vx0 = self.label_dict[ref[0]]
                vx1 = self.label_dict[ref[1]]
            else:
                print "cannot find labels %s" %ref
                return None
            key = [vx0.idx,vx1.idx]
            key.sort()
            key = tuple(key)
            if shape == 'arc':
                if key in self.arc_graph:
                    return ([vx0,vx1], self.arc_graph[key])
            elif shape == 'line':
                if key in self.line_graph:
                    return ([vx0,vx1], self.line_graph[key])
            print "cannot find corresponding shape"
            return None
            
        
    
    def get_ge(self, vx0, vx1, shape, vp=None):
        if shape == 'l':
            graph = self.line_graph
        elif shape == 'a':
            graph = self.arc_graph
            
        key = (vx0.idx,vx1.idx)
        if key not in graph:
            key = (vx1.idx,vx0.idx)
            if key not in graph:
                return None

        if shape == 'a':
            return graph[key][0]
        return graph[key]

        
            
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
            
    def assign_labels(self, segments, max_dist=20):
        for vertex in self.vx_list:
            dist_list = [pt2pt_dist(vertex.loc,segment.center) for segment in segments]
            idx_list = np.argsort(dist_list)
            for idx in idx_list:
                if segments[idx].label != "" and dist_list[idx] < max_dist:
                    vertex.assign_label(segments[idx].label)
                    self.label_dict[segments[idx].label] = vertex
                    break

    
            

        