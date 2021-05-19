
f = open('dev.txt','r')

of = open('new_dev.txt','w')
for i in f.readlines() :
	i = i.replace('B-PER','B')
	i = i.replace('B-ORG','B')
	i = i.replace('B-LOC','B')
	i = i.replace('I-PER','I')
	i = i.replace('I-ORG','I')
	i = i.replace('I-LOC','I')
	print(i,end='',file=of)
f.close()
of.close()
