#!/usr/bin/python
import json

data = [ { 'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4, 'e' : 5 } ]

with open('ee.json',newline='') as jsonfile :
	ddd = json.load(jsonfile)

data2 = json.dumps(data)
print(data2)
print(ddd)
print(ddd["a"])

f = open('ex.json',newline='',mode='w')
json.dump(ddd, f,indent=4)
f.close()