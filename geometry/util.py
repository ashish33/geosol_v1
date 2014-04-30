'''
Created on Apr 15, 2014

@author: minjoon
'''

import numpy as np

def segment_line(line, start, end):
    line = np.array(line)
    pt0 = line[:2]
    pt1 = line[2:]
    new_pt0 = pt0 + start*(pt1-pt0)
    new_pt1 = pt0 + end*(pt1-pt0)
    return np.append(new_pt0,new_pt1)

def cir2pt_dist(cir, pt):
    pt = np.array(pt)
    dist = pt2pt_dist(cir[:2],pt)
    return np.abs(dist-cir[2])

# distance between line and point
def line2pt_dist(line, pt):
    pt = np.array(pt)
    mid = line_mid(line)
    vector = line_vector(line)
    normal = line_normal(line)
    par_dist = np.abs(np.dot(pt-mid, vector))
    per_dist = np.abs(np.dot(pt-mid, normal)) 
    line_len = pt2pt_dist(line[:2],line[2:])
    if par_dist < line_len/2:
        return per_dist
    else:
        return min(pt2pt_dist(pt,line[:2]),pt2pt_dist(pt,line[2:]))
    
    
def line_mid(line):
    line = np.array(line)
    pt0 = line[:2]
    pt1 = line[2:]
    return (pt0+pt1)/2.0

def line_vector(line):
    line = np.array(line)
    pt0 = line[:2]
    pt1 = line[2:]
    return (pt1-pt0)/np.linalg.norm(pt0-pt1)

def line_normal(line):
    vector = line_vector(line)
    return np.array([vector[1],-vector[0]])
    


def pt2pt_dist(pt0, pt1):
    x0, y0 = pt0
    x1, y1 = pt1
    return np.sqrt((x1-x0)**2+(y1-y0)**2)

def sim_circle(pt1, r1, pt2, r2):
    if r1 <= r2:
        tmp = np.array(pt2)
        pt2 = np.array(pt1)
        pt1 = tmp
        tmp = r2
        r2 = r1
        r1 = tmp
    else:
        pt1 = np.array(pt1)
        pt2 = np.array(pt2)
    
    # if they have no intersection, return 0
    d= np.linalg.norm(pt2-pt1)
    if r1 + r2 <= d:
        return 0
    
    # if one is contained in the other, smaller cicle / bigger circle
    if r1 >= d + r2:
        return (float(r2)/r1)**2
    
    
    s1 = (r1*r1-r2*r2)/(2*d) + d/2 
    s2 = np.abs(d-s1)
    if s1> r1 or s2 > r2:
        pass
    if d**2 + r2**2 > r1**2:
        t1 = np.arccos(s1/r1)
        t2 = np.arccos(s2/r2)
        A1 = (r1**2)*t1-s1*np.sqrt(r1**2-s1**2)
        A2 = (r2**2)*t2-s2*np.sqrt(r2**2-s2**2)
    else:
        t1 = np.arccos(s1/r1)
        t2 = np.pi - np.arccos(s2/r2)
        A1 = (r1**2)*t1-s1*np.sqrt(r1**2-s1**2)
        A2 = (r2**2)*t2+s2*np.sqrt(r2**2-s2**2)
    ai = A1 + A2
    au = np.pi*r1**2 + np.pi*r2**2 - ai
    return ai/au

def sim_line(line1, line2):
    line1 = np.array(line1)
    line2 = np.array(line2)
    pt1 = np.dot((0.5,0.5),line1)
    r1 = np.linalg.norm(line1[1]-line1[0])/2
    pt2 = np.dot((0.5,0.5),line2)
    r2 = np.linalg.norm(line2[1]-line2[0])/2
    metric_dist = sim_circle(pt1,r1,pt2,r2)
    
    v1 = line1[1]-line1[0]
    v1 = v1/np.linalg.norm(v1)
    v2 = line2[1]-line2[0]
    v2 = v2/np.linalg.norm(v2)
    metric_angle = np.abs(np.dot(v1,v2))
    
    return metric_dist * metric_angle