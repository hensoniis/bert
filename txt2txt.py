#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import jieba
import os
import sys
import json

from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

def ckiptagger_fun_init():
	global ws,pos,ner
	data_path = "/media/henson/07ee3679-2a14-4401-a3b0-c767035ec0b6/user01/data"
	ws = WS(data_path, disable_cuda=False)
	pos = POS(data_path, disable_cuda=False)
	ner = NER(data_path, disable_cuda=False)

def ckiptagger_fun(sentence_list:list)->list:
	word_sentence_list = ws(sentence_list)
	# word_sentence_list = ws(sentence_list, sentence_segmentation=True)
	# word_sentence_list = ws(sentence_list, recommend_dictionary=dictionary)
	# word_sentence_list = ws(sentence_list, coerce_dictionary=dictionary)
	pos_sentence_list = pos(word_sentence_list)
	entity_sentence_list = ner(word_sentence_list, pos_sentence_list)

	def print_word_pos_sentence(word_sentence, pos_sentence)->list:
		assert len(word_sentence) == len(pos_sentence)
		_tmp_ = []
		for word, pos in zip(word_sentence, pos_sentence):
			_tmp_.append(f"{word}({pos})")
		return	_tmp_
	
	tmp = [
		sentence_list[0]
		,print_word_pos_sentence(word_sentence_list[0],  pos_sentence_list[0])
		,entity_sentence_list[0]
	]
	return tmp


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
	f = open("./test/out_put.json", mode='w')
	
	op_dict = {"WS":[],"POS":[],"NER":[]}
	
	
	for i in open("./test/input_data.txt").read().splitlines() :
		for j in i.split('，'):
			output_list = ckiptagger_fun([j])
			#print(f'{output_list[0]}\t{output_list[1]}\t{output_list[2]}',file=f)
			
			op_dict["WS"].append(output_list[0])
			op_dict["POS"].append(output_list[1])
			
			#j=0
			#for i in op_dict :
			#	op_dict[i].append(output_list[j])
			#	j += 1
			
			#op_dict = {"WS":output_list[0],"POS":output_list[1],"NER":output_list[2]}
	print(type(op_dict))
	json.dump([op_dict],f,indent=4)
	#print(opdata)
	f.close()
	
	with open("./test/out_put.json", mode='r') as jsonfile :
		data = json.load(jsonfile)
		print(data)
	ckiptagger_fun_clear()

############################33
if __name__ == '__main__':
	main()
	sys.exit()













