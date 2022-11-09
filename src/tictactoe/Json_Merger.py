import json
import ast
import numpy as np

data1 = {}
with open('table1.json', 'r') as fp:
	data = json.load(fp)
	for key, value in data.items():
		key = ast.literal_eval(key)
		value = np.array(value)
		data1[key] = value

data2 = {}
with open('table2.json', 'r') as fp:
	data = json.load(fp)
	for key, value in data.items():
		key = ast.literal_eval(key)
		value = np.array(value)
		data2[key] = value

data1.update(data2)

json_table = {}
for key, value in data1.items():
	json_table[str(key)] = value.tolist()
	
with open('tablemerged.json', 'w') as fp:
	json.dump(json_table, fp, indent=4)