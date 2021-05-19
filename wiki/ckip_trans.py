#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from ckip_transformers import __version__
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker

def main():

	# Show version
	print(__version__)

	# Initialize drivers
	#print('Initializing drivers ... WS')
	ws_driver  = CkipWordSegmenter(level=3)
	#print('Initializing drivers ... POS')
	#pos_driver = CkipPosTagger(level=3)
	#print('Initializing drivers ... NER')
	#ner_driver = CkipNerChunker(level=3)
	#print('Initializing drivers ... done')
	#print()

	# Use GPU:0
	ws_driver = CkipWordSegmenter(device=0)

	# Input text
	
	input_data = open('output_data.txt','r')
	
	#text = input_data.readline()
	#text = [text.replace('\n','')]
	#print(text)
	
	
	# Run pipeline
	#print('Running pipeline ... WS')
	#ws  = ws_driver(text)
	#print(ws)
	
		#print(ws)
	tmp = []
	for text in input_data.readlines()[:16]: #由此控制轉換數量
		tmp.append(text.replace('\n',''))
	
	#print(tmp)
	ws = ws_driver(tmp)
	#print(ws)
	for i in ws :
		for j in i :
			if len(j) > 1 :
				step = 0
				for _tmp_ in j :
					if step == 0 :
						output_result.write(_tmp_+' '+'B'+'\n')
					else :
						output_result.write(_tmp_+' '+'I'+'\n')
					step += 1
			else :
				output_result.write(j+' '+'O'+'\n')
		output_result.write('\n')
	
	
	input_data.close()
	#print('Running pipeline ... POS')
	#pos = pos_driver(ws)
	#print('Running pipeline ... NER')
	#ner = ner_driver(text)
	#print('Running pipeline ... done')
	#print()
	'''
	# Show results
	for sentence, sentence_ws, sentence_pos, sentence_ner in zip(text, ws, pos, ner):
		print(sentence)
		print(pack_ws_pos_sentece(sentence_ws, sentence_pos))
		for entity in sentence_ner:
			print(entity)
		print()
	'''

# Pack word segmentation and part-of-speech results
def pack_ws_pos_sentece(sentence_ws, sentence_pos):
	assert len(sentence_ws) == len(sentence_pos)
	res = []
	for word_ws, word_pos in zip(sentence_ws, sentence_pos):
		res.append(f'{word_ws}({word_pos})')
	return '\u3000'.join(res)

if __name__ == '__main__':
	output_result = open('ws_result_16.txt','w')
	main()
	output_result.close()
