'''
Created on Apr 29, 2014

@author: minjoon
'''
from abc import ABCMeta, abstractmethod


class VisualElement:
    __metaclass__ = ABCMeta
    @abstractmethod
    def __repr__(self):
        pass
    
    
class VELine(VisualElement):
    def __init__(self, vp_line, start=0, end=1):
        self.vp_list = [vp_line]
