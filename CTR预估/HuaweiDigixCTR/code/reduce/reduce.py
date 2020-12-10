'''
Author: Kevin
Date: 2020-12-09 14:35:31
LastEditTime: 2020-12-09 15:49:24
LastEditors: Please set LastEditors
Description: Data Compression
FilePath: \HuaweiDigixCTR\code\reduce\reduce.py
'''
import pandas as pd
import numpy as np
from tqdm import tqdm


def reduce(dataset):
    int_list = ['int', 'int16', 'int32']
	float_list = ['float', 'float32']
    for col in tqdm(dataset.columns):
        col_type = dataset[col].dtypes
		if col_type in int_list:
			c_min = dataset[col].min()
			c_max = dataset[col].max()
			if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
				dataset[col] = dataset[col].astype(np.int8)
			elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
				dataset[col] = dataset[col].astype(np.int16)
			elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
				dataset[col] = dataset[col].astype(np.int32)
		elif col_type in float_list:
			c_min = dataset[col].min()
			c_max = dataset[col].max()
			if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
				dataset[col] = dataset[col].astype(np.float16)
			elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
				dataset[col] = dataset[col].astype(np.float32)
    return dataset

def reduce_s(dataset):
    int_list = ['int','int32','int16']
    float_list  = ['float','float32']
    col_type = dataset.dtypes
    if col_type in int_list:
		c_min = dataset.min()
		c_max = dataset.max()
		if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
			dataset = dataset.astype(np.int8)
		elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
			dataset = dataset.astype(np.int16)
		elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
			dataset = dataset.astype(np.int32)
	elif col_type in float_list:
		c_min = dataset.min()
		c_max = dataset.max()
		if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
			dataset = dataset.astype(np.float16)
		elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
			dataset = dataset.astype(np.float32)
	return dataset

if __name__ == '__main__':
    print("开始压缩训练集")
    dataset = pd.read_csv('../data/train_data.csv',sep='|')
    dataset = reduce(dataset)
    dataset.to_pickle('../data/train_data.pkl')
    print("训练集压缩完成，开始压缩测试集")
    dataset = pd.read_csv('../data/test_data_A.csv',sep='|')
    dataset = reduce(dataset)
    dataset.to_pickle('../data/test_data_A.pkl')
    dataset = pd.read_csv('../data/test_data_B.csv',sep='|')
	dataset = reduce(dataset)
	dataset.to_pickle('../data/test_data_B.pkl')
    print("测试集压缩完成")
