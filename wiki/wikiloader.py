from gensim.test.utils import datapath
from gensim.corpora import WikiCorpus


wiki = WikiCorpus("zhwiki-20210120-pages-articles-multistream1.xml-p1p187712.bz2",dictionary={})

f = open('output_data.txt','w')

tmp_list = []


def is_chinese(cp):
	''' Check if `cp` is a chinese character. '''
	cp = ord(cp)
	if ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF)
			or (0x20000 <= cp <= 0x2A6DF) or (0x2A700 <= cp <= 0x2B73F)
			or (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF)
			or (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F)):
		return True
	return False




max_test_len = 0
for text in wiki.get_texts():
	str_line = list(text)
	for i in str_line :
		res = True
		for _tmp_letter_ in i :
			if not is_chinese(_tmp_letter_) :
				res = False
				continue
		if max_test_len < len(i) :
			max_test_len = len(i)
		if res and len(i)>=10:
			f.write(i+'\n')

print(max_test_len)
f.close()
