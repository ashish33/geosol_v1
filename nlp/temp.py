'''
Created on May 14, 2014

@author: minjoon
'''
import re

SHAPE_KEYWORDS = ['line', 'circle', 'arc', 'chord', 'tangent', 'secant', 'triangle', 'angle', 'diameter']
PUNCT = '.,;'

class Word:
    def __init__(self, word, kind='normal'):
        self.word = word
        self.kind = kind
        if kind=='normal' and word in PUNCT:
            self.kind = 'punc'
        
    def set_word(self, word):
        self.word = word
        
    def set_kind(self, kind):
        self.kind = kind
        
    def __repr__(self):
        return "%s(%s)" %(self.word,self.kind)

    def html(self):
        if self.kind == 'normal' or self.kind == 'punc':
            return self.word
        elif self.kind == 'keyword':
            return "<a onclick=\"hclick('%s');\" href=\"javascript:void(0)\">%s</a>" %(self.word,self.word)
        
def is_ref(word):
    for char in word:
        if not char.isupper():
            return False
    return True

def is_shape(word):
    if word.lower() in SHAPE_KEYWORDS:
        return True
    return False

# determine if a list of words is a keyword    
def is_keyword(words):
    if len(words) == 1:
        if is_ref(words[0]):
            return True
    elif len(words) == 2:
        if is_shape(words[0]) and is_ref(words[1]):
            return True
    return False

def split_sentence(sentence):
    
    return [word for word in re.split('(\W)', sentence) if word not in [" ", ""]]

def generate_word_list(sentence):
    str_list = split_sentence(sentence)
    word_list = []
    # first create a list of Word
    # second replace length-2 keyword
    idx = 0
    
    while idx < len(str_list)-1:
        str0 = str_list[idx]
        str1 = str_list[idx+1]
        if is_keyword([str0,str1]):
            word = Word(str0 + ' ' + str1, 'keyword')
            word_list.append(word)
            idx += 2
        elif is_keyword([str0]):
            word = Word(str0, 'keyword')
            word_list.append(word) 
            idx += 1
        else:
            word_list.append(Word(str0))
            idx += 1
    if idx < len(str_list):
        if is_keyword([str_list[idx]]):
            word_list.append(Word(str_list[idx], 'keyword'))
        else:
            word_list.append(Word(str_list[idx]))
    
    return word_list

def repr_words(words, fn="word"):
    print fn
    word = words[0]
    out = eval('word.'+fn)
    for word in words[1:]:
        if word.kind != 'punc':
            out += " "
        cmd = "word." + fn
        out += eval(cmd)
    return out
    

def test_generate_word_list():
    sentence = 'Circle O has a radius of 5. Line AB is 5. Find AD.'
    words = generate_word_list(sentence)
    print repr_words(words, 'html()')

if __name__ == '__main__':
    test_generate_word_list()

        
