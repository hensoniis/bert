import os
import sys
import torch
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForQuestionAnswering, BertForSequenceClassification, BertModel

import torch.nn as nn

#from QANet import CQAttention

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
