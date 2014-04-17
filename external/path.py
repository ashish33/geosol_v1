'''
Created on Apr 16, 2014

@author: minjoon
'''
import os.path
import sys

def directory_iterator(path, digit, extension='', num=0):
    dir_list = os.listdir(path)
    if len(extension) > 0:
        extension = '.' + extension
    name = ('{0:0%d}'%digit).format(num) + extension
    folder_list = []
    while name in dir_list:
        folder_list.append(os.path.join(path,name))
        num += 1
        name = ('{0:0%d}'%digit).format(num) + extension
    return folder_list

if __name__ == '__main__':
    print directory_iterator(sys.argv[1], 6, 'csv')
    
def find_file(path, name):
    dir_list = os.listdir(path)
    for filename in dir_list:
        if name == filename.split('.')[0]:
            return os.path.join(path, filename)
    return None

def next_name(path, digit, extension='', num=0):
    dir_list = os.listdir(path)
    if len(extension) > 0:
        extension = '.' + extension
    num = 0
    name = ('{0:0%d}'%digit).format(num) + extension
    while name in dir_list:
        num += 1
        name = ('{0:0%d}'%digit).format(num) + extension
    return os.path.join(path,name)
    