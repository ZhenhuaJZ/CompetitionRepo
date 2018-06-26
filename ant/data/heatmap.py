import numpy as np
import pandas as pd

def splitme_zip(a,d):
	m = np.concatenate(([True], a[1:]>a[:-1] + d, [True]))
	idx = np.flatnonzero(m)
	l = a.tolist()
	return [l[i:j] for i,j in zip(idx[:-1], idx[1:])]

corr_path = "heat_map.csv"
df = pd.read_csv(corr_path)
total_index_y = []
total_index_x = []
_row_index_list = []
for i in range(len(df)):
	df = pd.read_csv(corr_path)
	col_name = df.columns.values.tolist()[1:]
	#print(col_name)
	#df_y = df.drop([i], axis=0)
	#print(col_name[i])
	df_x = df.drop([col_name[i]], axis =1)
	#print(df_x)
	#v1_y = df_y.iloc[:,i+1]
	v1_x = df_x.iloc[i,:]
	#index_y = v1_y.index.values
	index_x = df_x.columns.values.tolist()
	row_index = df_x.index.values[i]
	#_row_index ="f{}".format(row_index+1)
	_row_index = row_index+1
	#y_index = [i for i, v in zip(index_y, v1_y) if v ==1 or v == 0.9]
	x_index = [i for i, v in zip(index_x, v1_x) if v ==1]
	#new_heat.append([1 if v ==1 or v == 0.9 else 0 for i, v in zip(index, v1)])

	#if len(y_index) > 0:
		#total_index_y.append(y_index)
	if len(x_index) > 0:
		#print(_row_index)
		#print(x_index)
		_row_index_list.append(_row_index)
		total_index_x.append(x_index)

key_segement = splitme_zip(np.array(_row_index_list), 1)
dictionary = dict(zip(_row_index_list, total_index_x))

seg_dict = []
t_seg_dict = []
for key in key_segement:
	for i in key:
		seg_dict.extend(dictionary[i]) #concatenet list
	seg_dict = [int(w.replace('f', '')) for w in seg_dict] #delete "f"
	seg_dict = list(set(seg_dict)) #delete duplicate
	if len(seg_dict) > 1:
		t_seg_dict.append(seg_dict)
	seg_dict = []
#print(t_seg_dict)
f_l =[]
for seg in t_seg_dict:
	segement = splitme_zip(np.array(seg), 1)
	#print(segement)
	l = []
	for i, s in enumerate(segement):
		if len(s) < 4:
			l.append(i)

	_segement = [i for j, i in enumerate(segement) if j not in l]
	l2 = []
	for i in _segement:
		if len(i)>0:
			l2.extend(i)
	f_l.extend(l2)

final = list(set(f_l))

f_segement = splitme_zip(np.array(final), 1)

delet_l = []
for l in f_segement:
	l.pop(0)
	delet_l.extend(l)

print(delet_l)
_delet_l = ['f'+str(w) for w in delet_l]
np.save("heat_map_del.npy", np.array(_delet_l))

    
#print(total_index_y)
#print(total_index_x)
#print(len(total_index_y))
#print(len(total_index_x))
final_drop_list = drop_list[0]
#new_heat = np.array(new_heat)
#new_heat = np.transpose(new_heat)
#np.savetxt("heat_map_2.csv", new_heat, delimiter = ',')

for i in range(1, len(drop_list)):
	#print(drop_list[i])
	final_drop_list.extend(x for x in drop_list[i] if x not in final_drop_list)
final_drop_list.sort()
#print(final_drop_list)






#split_l2 = splitme_zip(np.array(final_drop_list), 2)   

def drop_cor_features():

	split_l = splitme_zip(np.array(final_drop_list), 1)
	#print(split_l)
	new_split = []
	for i in range(len(split_l)):
		if len(split_l[i]) > 1:
			split_l[i].remove(split_l[i][0])
			for i in split_l[i]:
				new_split.append("f{}".format(i+1))

	#print(new_split)

drop_cor_features()