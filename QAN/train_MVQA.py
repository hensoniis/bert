import os
import sys
import torch
#from torch.optim import AdamW

from torch.optim import Adam	#pytorch版本較低

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForQuestionAnswering, BertForSequenceClassification, BertModel

import torch.nn as nn

# from QANet import CQAttention

import numpy as np
from os.path import join

from collections import Counter

import string
import re
import argparse

import pickle
import jieba

import json
from copy import deepcopy
from opencc import OpenCC
from tqdm import tqdm


import pdb

cc = OpenCC('tw2sp')
'''
if len(sys.argv) == 1:
	exit(1)
else:
	device = torch.device(sys.argv[1])
'''


log_file = open("log_file.txt","w")	#紀錄訓練過程



device = torch.device('cuda:0')

tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

# tokenizer = BertTokenizer.from_pretrained('roberta_base_lm_finetune')

# tokenizer = BertTokenizer.from_pretrained('roberta_large_lm_finetune')


norm_tokenizer = tokenizer

def is_chinese(cp):
	''' Check if `cp` is a chinese character. '''
	cp = ord(cp)
	if ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF)
			or (0x20000 <= cp <= 0x2A6DF) or (0x2A700 <= cp <= 0x2B73F)
			or (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF)
			or (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F)):
		return True
	return False

def tokenize_no_unk(tokenizer, text):
	''' Get BERT tokens without [UNK]. '''
	split_tokens = []
	for token in tokenizer.basic_tokenizer.tokenize(text, never_split=tokenizer.all_special_tokens):
		wp_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
		if wp_tokens == [tokenizer.unk_token]:
			split_tokens.append(token)
		else:
			split_tokens.extend(wp_tokens)
	return split_tokens

def bert_glove_match(glove_text, bert_text):
	macthed_glove_text = []
	same_tk_flag = False
	bert_c = 0
	# pdb.set_trace()
	for word in glove_text:
		if word == '\n' or word == ' ':
			continue
		token_r = ''
		for token in bert_text:
			if len(token_r) == len(word):
				# pdb.set_trace()
				token_r = ''
				break
			elif len(token) > 2:
				if token[:2] == '##':
					same_tk_flag = True
					macthed_glove_text.append(word)
					token_r += token[2:]
					bert_c += 1
				else:
					same_tk_flag = True
					macthed_glove_text.append(word)
					token_r += token
					bert_c += 1
			else:
				same_tk_flag = True
				macthed_glove_text.append(word)
				token_r += token
				bert_c += 1
		bert_text = bert_text[bert_c:]
		bert_c = 0
	return macthed_glove_text



def find_sublist(a, b, order=-1):
	''' Find the `order`-th sublist `b` in `a`. '''
	if not b: 
		return -1
	counter = 0
	for i in range(len(a)-len(b)+1):
		if a[i:i+len(b)] == b:
			counter += 1
			if counter > order:
				return i
	return -1

def convert_tokens_to_string(tokens):
	''' Recover Chinese BERT wordpiece tokens to string. '''
	output_string = str()
	ends_with_chinese = True
	for t in tokens:
		if is_chinese(t[-1]):
			output_string += t
			ends_with_chinese = True
		else:
			if ends_with_chinese:
				output_string += t
			else:
				output_string += ' ' + t
			ends_with_chinese = False

	output_string = output_string.replace(' ##', '')
	return output_string

def normalize_answer(s):
	"""Lower text and remove punctuation, articles and extra whitespace."""
	def remove_articles(text):
		return re.sub(r'\b(a|an|the)\b', ' ', text)

	def white_space_fix(text):
		return ' '.join(text.split())

	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)

	def lower(text):
		return text.lower()

	return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
	prediction_tokens = normalize_answer(prediction).split()
	ground_truth_tokens = normalize_answer(ground_truth).split()
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(prediction_tokens)
	recall = 1.0 * num_same / len(ground_truth_tokens)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1


def exact_match_score(prediction, ground_truth):
	return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
	scores_for_ground_truths = []
	for ground_truth in ground_truths:
		score = metric_fn(prediction, ground_truth)
		scores_for_ground_truths.append(score)
	return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
	f1 = exact_match = total = 0
	for article in dataset:
		for paragraph in article['paragraphs']:
			for qa in paragraph['qas']:
				total += 1
				if qa['id'] not in predictions:
					message = 'Unanswered question ' + qa['id'] + \
							  ' will receive score 0.'
					print(message, file=sys.stderr)
					continue
				ground_truths = list(map(lambda x: x['text'], qa['answers']))
				prediction = predictions[qa['id']]
				exact_match += metric_max_over_ground_truths(
					exact_match_score, prediction, ground_truths)
				f1 += metric_max_over_ground_truths(
					f1_score, prediction, ground_truths)

	exact_match = 100.0 * exact_match / total
	f1 = 100.0 * f1 / total

	return {'exact_match': exact_match, 'f1': f1}

def covert_seq2id(seq,table):
	out_seq = []
	for token in seq:
		if token not in table:
			out_seq.append(0)
		else:
			out_seq.append(table[token])
	return out_seq


class FGCDataset(Dataset):
	def __init__(self, split, datapath_list, bwd=False):

		assert split in ('train', 'dev', 'test')
		self.split = split
		self.bwd = bwd   # Pad inputs backwardly
		self.datas = []

		for datapath in datapath_list:
			data_file = open(datapath)
			input_data = json.load(data_file)

			passage_count = len(input_data)
			all_questions = 0
			amode_questions = 0
			impossible_questions = 0

			if datapath.startswith('movieqa'):
				DTEXT_KEY = 'DTEXT_CN'
				QTEXT_KEY = 'QTEXT_CN'
				ATEXT_KEY = 'ATEXT_CN'
				AMODE_KEY = 'AMODE_'
			else:
				DTEXT_KEY = 'DTEXT'
				QTEXT_KEY = 'QTEXT'
				ATEXT_KEY = 'ATEXT'
				AMODE_KEY = 'AMODE'


			for PQA in input_data:

				# Process passage
				raw_passage = PQA[DTEXT_KEY].strip()
				if len(raw_passage) == 0:
					raw_passage = '无文本'
					continue

				passage = tokenizer.tokenize(raw_passage)
				passage_no_unk = tokenize_no_unk(tokenizer, raw_passage)
				# pdb.set_trace()

				PID = PQA['DID']

				# QA pairs
				QAs, Q_order = [], 0
				for QA in PQA['QUESTIONS']:
					all_questions += 1
					if 'ANSWER' not in QA or \
					   QA[AMODE_KEY] != 'Single-Span-Extraction' and \
					   'Single-Span-Extraction' not in QA[AMODE_KEY]:
						continue
					amode_questions += 1
	 
					# Process question
					raw_question = QA[QTEXT_KEY].strip()
					if len(raw_question) == 0:
						raw_question = '无问题'
						continue
					question = tokenizer.tokenize(raw_question)
					question_no_unk = tokenize_no_unk(tokenizer, raw_question)


					# Find out N where the N-th occurence of a string is the answer
					raw_answers = [A[ATEXT_KEY].strip() for A in QA['ANSWER']]
					# pdb.set_trace()
					if len(raw_answers) == 0 or len(raw_answers[0]) == 0:
						raw_answers = ['[UNK]']
					if 'ATOKEN' in QA['ANSWER'][0]:
						raw_answer_start = QA['ANSWER'][0]['ATOKEN'][0]['start']
						found_answer_starts = [m.start() for m in re.finditer(raw_answers[0], raw_passage)]
						answer_order, best_dist = -1, 10000
						for order, found_start in enumerate(found_answer_starts):
							dist = abs(found_start - raw_answer_start)
							if dist < best_dist:
								best_dist = dist
								answer_order = order
					else:
						answer_order = -1

					# Find answer in the passage (which positions is the closest to the label)
					answer_no_unk = tokenize_no_unk(tokenizer, raw_answers[0])
					answer_start = find_sublist(passage_no_unk, answer_no_unk, order=answer_order)
					answer_end = answer_start + len(answer_no_unk) - 1 if answer_start >= 0 else -1
					if answer_start < 0:
						impossible_questions += 1

					# Store data (except bad data)
					# if answer_start >= 0 or split != 'train':
					if answer_start >= 0:
						processed_QA = {}
						processed_QA['question'] = question
						processed_QA['question_no_unk'] = question_no_unk
						processed_QA['answer'] = raw_answers
						processed_QA['answer_start'] = answer_start
						processed_QA['answer_end'] = answer_end
						processed_QA['id'] = '%s_%d' % (QA['QID'], Q_order)
						QAs.append(processed_QA)
						Q_order += 1


					for QA in QAs:
						input_item = {}
						question_id = QA['id']
						qtext_tokens = QA['question']
						qtext_no_unk_tokens = QA['question_no_unk']
						dtext_tokens = passage
						dtext_no_unk_tokens = passage_no_unk
						answer = QA['answer']

						qtext_tokens.insert(0, tokenizer.cls_token)
						qtext_tokens.append(tokenizer.sep_token)
						qtext_no_unk_tokens.insert(0, tokenizer.cls_token)
						qtext_no_unk_tokens.append(tokenizer.sep_token)

						answer_start = QA['answer_start'] + len(qtext_tokens)
						answer_end = QA['answer_end'] + len(qtext_tokens)

						# Truncate length to 512
						diff = len(qtext_tokens) + len(dtext_tokens) - 511


						if diff > 0:
							# Padding forwardly/backwardly is automatically decided for training
							if self.split == 'train':
								if answer_end > 510:
									answer_start -= diff
									answer_end -= diff
									dtext_tokens = dtext_tokens[diff:]
									dtext_no_unk_tokens = dtext_no_unk_tokens[diff:]
								else:
									dtext_tokens = dtext_tokens[:-diff]
									dtext_no_unk_tokens = dtext_no_unk_tokens[:-diff]
							else:
								if self.bwd:
									dtext_tokens = dtext_tokens[diff:]
									dtext_no_unk_tokens = dtext_no_unk_tokens[diff:]
								else:
									dtext_tokens = dtext_tokens[:-diff]
									dtext_no_unk_tokens = dtext_no_unk_tokens[:-diff]

						dtext_tokens.append(tokenizer.sep_token)
						dtext_no_unk_tokens.append(tokenizer.sep_token)

						input_tokens = qtext_tokens + dtext_tokens
						input_no_unk_tokens = qtext_no_unk_tokens + dtext_no_unk_tokens

						input_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(input_tokens))
						attention_mask = torch.FloatTensor([1 for _ in input_tokens])
						token_type_ids = torch.LongTensor([0 for _ in qtext_tokens] + [1 for _ in dtext_tokens])
						start_positions = torch.LongTensor([answer_start]).squeeze(0)
						end_positions = torch.LongTensor([answer_end]).squeeze(0)
						margin_mask = torch.FloatTensor([*(-1e10 for _ in qtext_tokens), *(0. for _ in dtext_tokens[:-1]), -1e-10])


						input_item['input_ids'] = input_ids
						input_item['attention_mask'] = attention_mask
						input_item['token_type_ids'] = token_type_ids
						input_item['start_positions'] = start_positions
						input_item['end_positions'] = end_positions
						input_item['input_no_unk_tokens'] = input_no_unk_tokens
						input_item['answer'] = answer
						input_item['margin_mask'] = margin_mask

						self.datas.append(input_item)

	def __len__(self):
		return len(self.datas)
		
	def __getitem__(self, i):
		input_ids = self.datas[i]['input_ids']
		attention_mask = self.datas[i]['attention_mask']
		token_type_ids = self.datas[i]['token_type_ids']
		start_positions = self.datas[i]['start_positions']
		end_positions = self.datas[i]['end_positions']
		input_no_unk_tokens = self.datas[i]['input_no_unk_tokens']
		answer = self.datas[i]['answer']
		margin_mask = self.datas[i]['margin_mask']

		if self.split == 'train':   # Answer for training
			return input_ids, attention_mask, token_type_ids, start_positions, end_positions
		else:   # Mask out question tokens for inference
			return input_ids, attention_mask, token_type_ids, margin_mask, input_no_unk_tokens, answer


def get_FGC_dataloader(split, bwd=False, batch_size=1, num_workers=0, datapath_list=None):
	''' Build dataloader with specified configs. '''
	def train_collate_fn(batch):
		input_ids, attention_mask, token_type_ids, start_positions, end_positions = zip(*batch)
		input_ids = pad_sequence(input_ids, batch_first=True)
		attention_mask = pad_sequence(attention_mask, batch_first=True)
		token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=1)
		start_positions = torch.stack(start_positions)
		end_positions = torch.stack(end_positions)
		return input_ids, attention_mask, token_type_ids, start_positions, end_positions
	
	def test_collate_fn(batch):
		input_ids, attention_mask, token_type_ids, margin_mask, input_tokens_no_unk, answers = zip(*batch)
		input_ids = pad_sequence(input_ids, batch_first=True)
		attention_mask = pad_sequence(attention_mask, batch_first=True)
		token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=1)
		margin_mask = pad_sequence(margin_mask, batch_first=True, padding_value=1e-10)
		return input_ids, attention_mask, token_type_ids, margin_mask, input_tokens_no_unk, answers

	shuffle = (split == 'train' or split == 'train_ranker')
	collate_fn = train_collate_fn if (split == 'train' or split == 'train_ranker') else test_collate_fn

	dataset = FGCDataset(split, datapath_list, bwd)

	dataloader = DataLoader(dataset, collate_fn=collate_fn, shuffle=shuffle, \
							batch_size=batch_size, num_workers=num_workers)
	return dataloader




# def f1_score(pred, answer):
#	 overlap = len(set(pred) & set(answer))
#	 if overlap == 0 or len(pred) == 0 or len(answer) == 0:
#		 return 0.0
#	 precision = overlap / len(pred)
#	 recall = overlap / len(answer)
#	 return 2 * precision * recall / (precision + recall)

# def exact_match(pred, answer):
#	 return 1 if pred == answer else 0

def validate_dataset(model, fwd_dataloader, bwd_dataloader, split, topk=1, datapath=None):
	''' Get F1/EM scores on dev/test split. '''
	assert split in ('dev', 'test')
	# Inference is based on both forwardly/backwardly padded inputs.
	# fwd_dataloader = get_FGC_dataloader(split, bwd=False, batch_size=16, num_workers=16, datapath=datapath)
	# bwd_dataloader = get_FGC_dataloader(split, bwd=True, batch_size=16, num_workers=16, datapath=datapath)
	em, f1, count = 0, 0, 0
	
	model.eval()
	for fwd_batch, bwd_batch in zip(fwd_dataloader, bwd_dataloader):
		##### Forward padding ####
		input_ids, attention_mask, token_type_ids, margin_mask, fwd_input_tokens_no_unks, answers = fwd_batch
		input_ids = input_ids.cuda(device=device)
		attention_mask = attention_mask.cuda(device=device)
		token_type_ids = token_type_ids.cuda(device=device)
		margin_mask = margin_mask.cuda(device=device)

		with torch.no_grad():
			outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

		# pdb.set_trace()
		
		# Mask out impossible tokens
		start_logits, end_logits = outputs[0], outputs[1]
		start_logits += margin_mask
		end_logits += margin_mask
		start_logits = start_logits.cpu().clone()
		fwd_end_logits = end_logits.cpu().clone()
		
		# Extract topk*5 start positions
		start_probs = start_logits
		fwd_start_probs, fwd_start_index = start_probs.topk(topk*5, dim=1)

		#### Backward padding ####
		input_ids, attention_mask, token_type_ids, margin_mask, bwd_input_tokens_no_unks, answers = bwd_batch
		input_ids = input_ids.cuda(device=device)
		attention_mask = attention_mask.cuda(device=device)
		token_type_ids = token_type_ids.cuda(device=device)
		margin_mask = margin_mask.cuda(device=device)

		with torch.no_grad():
			outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
		
		# Mask out impossible tokens
		start_logits, end_logits = outputs[0], outputs[1]
		start_logits += margin_mask
		end_logits += margin_mask
		start_logits = start_logits.cpu().clone()
		bwd_end_logits = end_logits.cpu().clone()

		# Extract topk*5 start positions
		start_probs = start_logits
		bwd_start_probs, bwd_start_index = start_probs.topk(topk*5, dim=1)

		#### Combine foward/backward predictions ####
		for i, answer in enumerate(answers):
			preds, probs = [], []
			for n in range(topk*5):
				#### Forward ####
				start_prob = fwd_start_probs[i][n].item()
				start_ind = fwd_start_index[i][n].item()
				beam_end_logits = fwd_end_logits[i].clone().unsqueeze(0)

				# Mask out impossible tokens
				end_probs = beam_end_logits
				end_probs[0, :start_ind] += -1e10
				end_probs[0, start_ind+20:] += -1e10
				end_probs, end_index = end_probs.topk(topk*5, dim=1)

				# Calculate probabilities of all (start+end) combinations
				for m in range(topk*5):
					end_prob = end_probs[0][m].item()
					end_ind = end_index[0][m].item()

					# Get prediction string and its probability 
					prob = start_prob + end_prob   # log prob (i.e. logits)
					span_tokens = fwd_input_tokens_no_unks[i][start_ind:end_ind+1]
					pred = convert_tokens_to_string(span_tokens)

					# Record each prediction string with its highest probability
					if pred == tokenizer.sep_token or pred == '':
						pass
					elif pred and pred not in preds:
						probs.append(prob)
						preds.append(pred)
					elif pred and pred in preds:
						pred_idx = preds.index(pred)
						if prob > probs[pred_idx]:
							probs[pred_idx] = prob
					else:
						pass
				
				#### Backward ####
				start_prob = bwd_start_probs[i][n].item()
				start_ind = bwd_start_index[i][n].item()
				beam_end_logits = bwd_end_logits[i].clone().unsqueeze(0)

				
				# Mask out impossible tokens
				end_probs = beam_end_logits
				end_probs[0, :start_ind] += -1e10
				end_probs[0, start_ind+20:] += -1e10
				end_probs, end_index = end_probs.topk(topk*5, dim=1)
				end_ind = end_index[0][0]

				# Calculate probabilities of all (start+end) combinations
				for m in range(topk*5):
					end_prob = end_probs[0][m].item()
					end_ind = end_index[0][m].item()

					# Get prediction string and its probability
					prob = start_prob + end_prob   # log prob (i.e. logits)
					span_tokens = bwd_input_tokens_no_unks[i][start_ind:end_ind+1]
					pred = convert_tokens_to_string(span_tokens)

					# Record each prediction string with its highest probability
					if pred == tokenizer.sep_token or pred == '':
						pass
					elif pred and pred not in preds:
						probs.append(prob)
						preds.append(pred)
					elif pred and pred in preds:
						pred_idx = pred.index(pred)
						if prob > probs[pred_idx]:
							probs[pred_idx] = prob
					else:
						pass

			count += 1
			if len(preds) > 0:
				# Extract topk predictions sorted by probabilities
				sorted_probs_preds = list(reversed(sorted(zip(probs, preds))))
				probs, preds = map(list, zip(*sorted_probs_preds))
				probs, preds = probs[:topk], preds[:topk]
				
				# Predictions and answers are normalized by BERT tokenizer for evaluating
				norm_preds_tokens = [norm_tokenizer.basic_tokenizer.tokenize(pred) for pred in preds]
				norm_preds = [norm_tokenizer.convert_tokens_to_string(norm_pred_tokens) for norm_pred_tokens in norm_preds_tokens]
				norm_answer_tokens = [norm_tokenizer.basic_tokenizer.tokenize(ans) for ans in answer]
				norm_answer = [norm_tokenizer.convert_tokens_to_string(ans_tokens) for ans_tokens in norm_answer_tokens]
			
				em += max(metric_max_over_ground_truths(exact_match_score, norm_pred, norm_answer) for norm_pred in norm_preds)
				f1 += max(metric_max_over_ground_truths(f1_score, norm_pred, norm_answer) for norm_pred in norm_preds)
			
	del fwd_dataloader, bwd_dataloader
	return em, f1, count

def validate(model, fwd_dataloader_d, bwd_dataloader_d, fwd_dataloader_t, bwd_dataloader_t, topk=1):
	''' Perform a verbose validation on both dev/test split. '''

	# Valid set
	val_em, val_f1, val_count = validate_dataset(model, fwd_dataloader_d, bwd_dataloader_d, 'dev', topk, 'movieqa_dev.json')
	val_avg_em = 100 * val_em / val_count
	val_avg_f1 = 100 * val_f1 / val_count

	# Test set
	test_em, test_f1, test_count = validate_dataset(model, fwd_dataloader_t, bwd_dataloader_t, 'test', topk, 'movieqa_test.json')
	test_avg_em = 100 * test_em / test_count
	test_avg_f1 = 100 * test_f1 / test_count
	
	print('%d-best | val_em=%.5f, val_f1=%.5f | test_em=%.5f, test_f1=%.5f' \
		% (topk, val_avg_em, val_avg_f1, test_avg_em, test_avg_f1))
	print('%d-best | val_em=%.5f, val_f1=%.5f | test_em=%.5f, test_f1=%.5f' \
		% (topk, val_avg_em, val_avg_f1, test_avg_em, test_avg_f1),file = log_file)	#紀錄
	return val_avg_f1

def validate_train(model, fwd_dataloader, bwd_dataloader, topk=1):
	''' Perform a verbose validation on both dev/test split. '''
	# Test set
	test_em, test_f1, test_count = validate_dataset(model, fwd_dataloader, bwd_dataloader, 'test', topk, 'movieqa_test.json')
	test_avg_em = 100 * test_em / test_count
	test_avg_f1 = 100 * test_f1 / test_count
	print('%d-best | test_em=%.5f, test_f1=%.5f' % (topk,  test_avg_em, test_avg_f1))
	print('%d-best | test_em=%.5f, test_f1=%.5f' % (topk,  test_avg_em, test_avg_f1),file = log_file)	#紀錄
	return test_avg_f1



def validate_dataset_ranker(ranker, fwd_dataloader, bwd_dataloader, split, topk=1, datapath=None):
	''' Get F1/EM scores on dev/test split. '''
	assert split in ('dev', 'test')
	# Inference is based on both forwardly/backwardly padded inputs.
	# fwd_dataloader = get_FGC_dataloader(split, bwd=False, batch_size=16, num_workers=16, datapath=datapath)
	# bwd_dataloader = get_FGC_dataloader(split, bwd=True, batch_size=16, num_workers=16, datapath=datapath)
	em, count = 0, 0
	
	ranker.eval()
	for fwd_batch, bwd_batch in zip(fwd_dataloader, bwd_dataloader):
		##### Forward padding ####
		input_ids, attention_mask, token_type_ids, _, _, _, inupt_glove_ids, inupt_w2v_ids, rank_truth = fwd_batch
		input_ids = input_ids.cuda(device=device)
		attention_mask = attention_mask.cuda(device=device)
		token_type_ids = token_type_ids.cuda(device=device)
		rank_truth = rank_truth.cuda(device=device)

		with torch.no_grad():
			fwd_out = ranker(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
			fwd_out = fwd_out[0]
			fwd_out = torch.nn.functional.softmax(fwd_out, -1)
		# pdb.set_trace()
		prediction = torch.argmax(fwd_out,dim=-1)
		rank_truth = rank_truth.squeeze()
		count += input_ids.size(0)
		em += torch.sum(prediction ^ rank_truth).item()

		#### Backward padding ####
		input_ids, attention_mask, token_type_ids, _, _, _, inupt_glove_ids, inupt_w2v_ids, rank_truth = bwd_batch
		input_ids = input_ids.cuda(device=device)
		attention_mask = attention_mask.cuda(device=device)
		token_type_ids = token_type_ids.cuda(device=device)
		rank_truth = rank_truth.cuda(device=device)

		with torch.no_grad():
			bwd_out = ranker(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
			bwd_out = bwd_out[0]
			bwd_out = torch.nn.functional.softmax(bwd_out, -1)
		# pdb.set_trace()
		prediction = torch.argmax(bwd_out,dim=-1)
		rank_truth = rank_truth.squeeze()
		count += input_ids.size(0)
		em += torch.sum(prediction ^ rank_truth).item()
						
	del fwd_dataloader, bwd_dataloader
	return em, count


def validate_ranker(ranker, fwd_dataloader_d, bwd_dataloader_d):
	''' Perform a verbose validation on both dev/test split. '''

	# Valid set
	val_em, val_count = validate_dataset_ranker(ranker, fwd_dataloader_d, bwd_dataloader_d, 'dev')
	val_avg_em = 100 * val_em / val_count

	return val_avg_em



if __name__ == '__main__':

	
	# Config
	d_model = 768
	dropout = 0.1
	lr=3e-5
	####batch_size = 4
	####accumulate_batch_size = 32
	batch_size = 4
	accumulate_batch_size = 32
	assert accumulate_batch_size % batch_size == 0
	update_stepsize = accumulate_batch_size // batch_size


	# model = BertForQuestionAnswering.from_pretrained('bert-base-chinese')
	model = BertForQuestionAnswering.from_pretrained('hfl/chinese-roberta-wwm-ext')
	# model = BertForQuestionAnswering.from_pretrained('roberta_base_lm_finetune')
	# model = BertForQuestionAnswering.from_pretrained('roberta_large_lm_finetune')

	model.to(device)
	#optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
	optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
	
	optimizer.zero_grad()

	em_dev_best = 0
	f1_dev_best = 0

	em_test_best = 0
	f1_test_best = 0

	f1_best = 0

	step = 0


	datapath_list_train = ['movieqa_train.json']
	#datapath_list_train = ['movieqa_train.json', 'DRCD_train.json', 'Kaggle_train.json', 'FGC_release_all_train.json', 'Lee_train.json']
	dataloader = get_FGC_dataloader('train', bwd=False, batch_size=batch_size, num_workers=8, datapath_list=datapath_list_train)

	####fwd_dataloader_tr = get_FGC_dataloader('dev', bwd=False, batch_size=16, num_workers=16, datapath_list=['movieqa_train.json'])
	####bwd_dataloader_tr = get_FGC_dataloader('dev', bwd=True, batch_size=16, num_workers=16, datapath_list=['movieqa_train.json'])

	####fwd_dataloader_d = get_FGC_dataloader('dev', bwd=False, batch_size=16, num_workers=16, datapath_list=['movieqa_dev.json'])
	####bwd_dataloader_d = get_FGC_dataloader('dev', bwd=True, batch_size=16, num_workers=16, datapath_list=['movieqa_dev.json'])
	####fwd_dataloader_t = get_FGC_dataloader('test', bwd=False, batch_size=16, num_workers=16, datapath_list=['movieqa_test.json'])
	####bwd_dataloader_t = get_FGC_dataloader('test', bwd=True, batch_size=16, num_workers=16, datapath_list=['movieqa_test.json'])

	fwd_dataloader_tr = get_FGC_dataloader('dev', bwd=False, batch_size=16, num_workers=16, datapath_list=['movieqa_train.json'])
	bwd_dataloader_tr = get_FGC_dataloader('dev', bwd=True, batch_size=16, num_workers=16, datapath_list=['movieqa_train.json'])

	fwd_dataloader_d = get_FGC_dataloader('dev', bwd=False, batch_size=16, num_workers=16, datapath_list=['movieqa_dev.json'])
	bwd_dataloader_d = get_FGC_dataloader('dev', bwd=True, batch_size=16, num_workers=16, datapath_list=['movieqa_dev.json'])
	fwd_dataloader_t = get_FGC_dataloader('test', bwd=False, batch_size=16, num_workers=16, datapath_list=['movieqa_test.json'])
	bwd_dataloader_t = get_FGC_dataloader('test', bwd=True, batch_size=16, num_workers=16, datapath_list=['movieqa_test.json'])


	for itr in tqdm(range(20)):

		for batch in dataloader:
			input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch

			input_ids = input_ids.cuda(device=device)
			attention_mask = attention_mask.cuda(device=device)
			token_type_ids = token_type_ids.cuda(device=device)
			start_positions = start_positions.cuda(device=device)
			end_positions = end_positions.cuda(device=device)

			model.train()
	
			tmp = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, \
				   start_positions=start_positions, end_positions=end_positions)


			#print(tmp[0])
			#print(type(tmp[0]))
			####print(type(loss))
			####sys.exit()
			tmp[0].backward()

			step += 1
			print("step %d | Training...\r" % step, end='')
	
			if step % update_stepsize == 0:
				optimizer.step()
				optimizer.zero_grad()


		print("step %d | Validating..." % step)
		print("step %d | Validating..." % step,file = log_file)	#紀錄
		_ = validate_train(model, fwd_dataloader_tr, bwd_dataloader_tr, topk=1)
		val_f1 = validate(model, fwd_dataloader_d, bwd_dataloader_d, fwd_dataloader_t, bwd_dataloader_t, topk=1)

		# em_dev_current, f1_dev_current, em_test_current, f1_test_current = validate(model)

		if f1_best < val_f1:
			torch.save(model.state_dict(), 'bestRoBERTaMVQA_full_dev_f1_mvqa0.pth')

			f1_best = val_f1
			print("f1_best %s" % f1_best)
			print("f1_best %s" % f1_best,file = log_file)	#紀錄
	
	log_file.close()

	
