import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library
import matplotlib.pyplot as plt
import time
from subprocess import check_output
from data_processing import *

#correlation map
def correlation_map():
	f,ax = plt.subplots(figsize=(25, 25))
	sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
	plt.show()

#path pd data frame and save_hist path
def hist_visualization(df_0, df_1, prefix, figure_1, figure_2):
	save_path = "../data/hist/" + prefix + "/"
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	#read data and choose column
	nbins = 80
	col_list = df_0.columns.values.tolist()
	for i in range(len(col_list)):
		col_name = col_list[i]
		print(col_name)
		fig, axes = plt.subplots(1,2, sharey =True, sharex = True, figsize=(35,15))
		feature_0 = df_0.loc[:,[col_name]]
		#Statics info
		mean = feature_0.mean().values[0]
		std = feature_0.std().values[0]
		most_frq_0 = feature_0.mode().values[0][0]
		most_mean_0 = round(mean,2)
		most_std_0 = round(std,2)
		most_quantile_low_0 = feature_0.quantile(q=0.25).values[0]
		most_quantile_high_0 = feature_0.quantile(q=0.75).values[0]
		feature_0 = (feature_0 - mean) / (std)
		feature_0.hist(ax = axes[0], bins= nbins,  xlabelsize= 25 , ylabelsize = 25 ,density = True)
		#feature_0.plot(ax = axes[0], color = 'green')
		#axes[0].plot(feature_0.index, feature_0.values, 'b.')
		#axes[0].tick_params(labelsize = 25)
		feature_1 = df_1.loc[:,[col_name]]
		mean = feature_1.mean().values[0]
		std = feature_1.std().values[0]
		most_frq_1 = feature_1.mode().values[0][0]
		most_mean_1 = round(feature_1.mean().values[0],2)
		most_std_1 = round(feature_1.std().values[0],2)
		most_quantile_low_1 = feature_1.quantile(q=0.25).values[0]
		most_quantile_high_1 = feature_1.quantile(q=0.75).values[0]
		feature_1 = (feature_1 - mean) / (std)
		feature_1.hist(ax = axes[1], bins= nbins, xlabelsize = 25 , ylabelsize = 25, density = True, color = 'r')
		#feature_1.plot(ax = axes[1], color = 'r')
		#axes[1].plot(feature_1.index, feature_1.values, 'r.')
		#axes[1].tick_params(labelsize = 25)
		#axes[1].hist(feature_1.values, bins = nbins, density =True, color = 'r')
		#plt.legend([axes[0], axes[1]], ["white, "black"], loc = "upper right")
		#Plot annotate
		axes[0].annotate("mode : " + str(most_frq_0), xy = (0.80,0.95), xycoords = 'axes fraction', size = 20)
		axes[1].annotate("mode : " + str(most_frq_1), xy = (0.80,0.95), xycoords = 'axes fraction' , size = 20)
		axes[0].annotate("mean : " + str(most_mean_0), xy = (0.80,0.92), xycoords = 'axes fraction', size = 20)
		axes[1].annotate("mean : " + str(most_mean_1), xy = (0.80,0.92), xycoords = 'axes fraction' , size = 20)
		axes[0].annotate("std : " + str(most_std_0), xy = (0.80,0.89), xycoords = 'axes fraction', size = 20)
		axes[1].annotate("std : " + str(most_std_1), xy = (0.80,0.89), xycoords = 'axes fraction' , size = 20)
		axes[0].annotate("q.25 : " + str(most_quantile_low_0), xy = (0.80,0.86), xycoords = 'axes fraction', size = 20)
		axes[1].annotate("q.25 : " + str(most_quantile_low_1), xy = (0.80,0.86), xycoords = 'axes fraction' , size = 20)
		axes[0].annotate("q.75 : " + str(most_quantile_high_0), xy = (0.80,0.83), xycoords = 'axes fraction', size = 20)
		axes[1].annotate("q.75 : " + str(most_quantile_high_1), xy = (0.80,0.83), xycoords = 'axes fraction' , size = 20)
		axes[0].set_title(col_name + str(figure_1), fontsize = 25)
		axes[1].set_title(col_name + str(figure_2), fontsize = 25)
		plt.savefig(save_path + "{}.png".format(col_name))
		plt.figure()


def time_stamp(date_column):
    for i in range(len(date_column)):
        if date_column.date[i] <= 20170915 and date_column.date[i+1] >= 20170916:
            time_stamp_1 =  i+1
        elif date_column.date[i] <= 20170930 and  date_column.date[i+1] >= 20171001:
            time_stamp_2 =  i+1
        elif date_column.date[i] <= 20171015 and  date_column.date[i+1] >= 20171016:
            time_stamp_3 =  i+1
        elif date_column.date[i] <= 20171031 and  date_column.date[i+1] >= 20171101:
            time_stamp_4 =  i+1
    return time_stamp_1, time_stamp_2, time_stamp_3, time_stamp_4

def save_features(data_path, prefix, mode = "train"):
	save_path = "../data/feats/" + prefix + "/"
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	#read data and choose column
	df = pd.read_csv(data_path)
	df = df.sort_values('date')
	date = df.loc[:,["date"]]

	for i in range(297):
		col_name = "f"+str(i+1)
		print(col_name)
		if mode == "train":
			print(df.loc[:,["date"]])
			df_0 = df[(df.label==0)]
			df_1 = df[(df.label==1)]
			feature_0 = df_0.loc[:,[col_name]]
			plt.plot(feature_0.index, feature_0.values, 'b.')
			feature_1 = df_1.loc[:,[col_name]]
			plt.plot(feature_1.index, feature_1.values, 'r.')
			plt.legend(("white", "black"), loc = 'upper right')
			t1, t2, t3, t4 = time_stamp(date)
			draw_time_stamp = lambda x,ymin,ymax: plt.vlines(x, ymin, ymax, linestyles = 'dashed')
			ymin, ymax = plt.ylim()
			for i in [t1, t2, t3, t4]:
				draw_time_stamp(i, ymin, ymax)

		elif mode == "test":
			feature = df.loc[:,[col_name]]
			#standarlization
			mean = feature.mean().values[0]
			std = feature.std().values[0]
			feature = (feature_1 - mean) / (std)
			#print(df.loc[:,["date"]])
			plt.plot(feature.index, feature.values, 'b.')
			plt.legend(("test_b"), loc = 'upper right')

		plt.ylabel('values')
		plt.xlabel("sample_number")
		plt.title(col_name)
		plt.savefig(save_path + "{}.png".format(col_name))
		plt.figure()

def main():

	train_path = "../data/train.csv"
	test_path = "../data/test_a.csv"

	_train_data = pd.read_csv(train_path)
	_test_online = pd.read_csv(test_path)

	_train_data, _test_offline =  test_train_split_by_date(_train_data, 20171010, 20171020)
	clear_mermory(_train_data)
	_test_online = _test_online.iloc[:,2:]
	_test_offline = _test_offline.iloc[:,3:]
	#hist_visualization(_train, _test_online, "train_test_b", figure_1="train", figure_2="test_b")
	#hist_visualization(_test_offline, _test_online, "on-off-1010-1020", figure_1="offline", figure_2="online")
	save_features(train_path, "std_feats", mode = "test")

if __name__ == '__main__':
	main()
