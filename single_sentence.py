#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import jieba
import os
import sys
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

def ckiptagger_fun_init():
    global ws,pos,ner
    data_path = "./data"
    ws = WS(data_path, disable_cuda=False)
    pos = POS(data_path, disable_cuda=False)
    ner = NER(data_path, disable_cuda=False)

def ckiptagger_fun(sentence_list:list):
    word_sentence_list = ws(sentence_list)
    # word_sentence_list = ws(sentence_list, sentence_segmentation=True)
    # word_sentence_list = ws(sentence_list, recommend_dictionary=dictionary)
    # word_sentence_list = ws(sentence_list, coerce_dictionary=dictionary)
    pos_sentence_list = pos(word_sentence_list)
    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)

    def print_word_pos_sentence(word_sentence, pos_sentence):
        assert len(word_sentence) == len(pos_sentence)
        for word, pos in zip(word_sentence, pos_sentence):
            print(f"{word}({pos})", end="\u3000")
        print()
        return
        
    for i, sentence in enumerate(sentence_list):
        print()
        print(f"'{sentence}'")
        print_word_pos_sentence(word_sentence_list[i],  pos_sentence_list[i])
        for entity in sorted(entity_sentence_list[i]):
            print(entity)
    return


def ckiptagger_fun_clear():
    global ws,pos,ner
    del ws
    del pos
    del ner



def jeiba_fun(_str_:str)->str:
    _mode_ = "True"
    seg_list = jieba.cut(_str_, cut_all=_mode_)
    return "/ ".join(seg_list)



def main():
    ckiptagger_fun_init()
    while True :
        str_input = []
        print("請輸入句子：")
        while True:
            tmp = str(input())
            if tmp != "" :
                str_input.append(tmp)
            else :
                break
        if len(str_input) == 0 :
            break
        ckiptagger_fun(str_input)
		
		

        print()
    ckiptagger_fun_clear()

############################33
if __name__ == '__main__':
    main()
    sys.exit()



