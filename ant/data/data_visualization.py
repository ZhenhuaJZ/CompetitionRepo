import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import time
from subprocess import check_output
#print(check_output(["ls", "./train_date.csv"]).decode("utf8"))

"""
data = pd.read_csv('./train_dated.csv')
data = data.iloc[:,:]
save_path = "ant_visualizaiton/"
col = data.columns       # .columns gives columns names in data
y = data.label                         # M or B
list = ['date','id','label']
x = data.drop(list,axis = 1 )
"""

"""
ax = sns.countplot(y,label="Count")
W, B = y.value_counts()
plt.savefig(save_path + "data_balance.png")
print('Number of White: ',W)
print('Number of Black : ',B)
"""

# first ten features
"""
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization

start = [64,72,76,102,107,111,155,161,166]
end = [71,75,101,106,110,154,160,165,210]
#[20,24,28,32,36,48,52,54]
#[23,27,31,35,47,51,53,63]

for s, e in zip(start, end):
	data = pd.concat([y,data_n_2.iloc[:,s-1:e-1]],axis=1)
	data = pd.melt(data,id_vars="label",
	                    var_name="features",
	                    value_name='value')
	plt.figure(figsize=(10,10))
	sns.violinplot(x="features", y="value", hue="label", data=data, split=True, inner="quart")
	plt.xticks(rotation=90)
	plt.savefig(save_path + "violin_plot_f{}_f{}.png".format(s, e))
	plt.show()
"""
"""
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="label",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(35,25))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="label", data=data)

plt.xticks(rotation=90)
plt.show()
"""
#correlation map

data = pd.read_csv('./train_heatmap.csv')

#save_corr
#_data = data.iloc[:,3:]
#print(_data.corr().values[33])
#print(_data.corr().values[34])
#_data.corr().to_csv('./corr_data/heat_map.csv')

start = [20,28,36,48,64,76,102,111,155,166,211,254,278]
end = [27,35,47,63,75,101,110,154,165,210,253,277,297]


for s, e in zip(start, end):
	_data = data.iloc[:,s+2:e+2]
	save_path = "ant_visualizaiton/"
	#col = data.columns       # .columns gives columns names in data
	#y = data.label                         # M or B
	#list = ['date','id','label']
	#x = data.drop(list,axis = 1 )
	f,ax = plt.subplots(figsize=(25,25))
	#print(_data.corr().values[0])
	
	sns.heatmap(_data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, annot_kws = {"size" : 20})
	plt.tick_params(axis='both', labelsize = '30')
	plt.savefig(save_path + "aft_heat_map_train_f{}_f{}.png".format(s, e))
	#plt.show()
