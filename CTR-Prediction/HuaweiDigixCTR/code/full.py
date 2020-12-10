'''
Author: Kevin
Date: 2020-12-09 15:45:35
LastEditTime: 2020-12-09 22:55:39
LastEditors: Please set LastEditors
Description: Training and reasoning of full feature model
FilePath: \HuaweiDigixCTR\code\full.py
'''

# here put the import lib
import pandas as pd
import numpy as np
from tqdm import tqdm


def adjust(df, key, feature):
    if 'key' == 'uid':
        mean7 = df[df['pt_d'] < 8][feature].mean()
        std = df[df['pt_d'] < 8][feature].std()
        mean8 = df[(df['pt_d'] >= 8) & (df['coldu'] == 1)][feature].mean()
        std8 = df[(df['pt_d'] >= 8) & (df['coldu'] == 1)][feature].std()
        df.loc[(df['pt_d'] >= 8) & (df['coldu'] == 1), feature] = (
            (df[(df['pt_d'] >= 8) & (df['coldu'] == 1)][feature] - mean8) /
            std8 * std7 + mean7)
    return df


# def adjust(df, key, feature):
# 	if key == 'uid':
# 		mean7 = df[df['pt_d'] < 8][feature].mean()
# 		std7 = df[df['pt_d'] < 8][feature].std()
# 		mean8 = df[(df['pt_d'] >= 8) & (df['coldu'] == 1)][feature].mean()
# 		std8 = df[(df['pt_d'] >= 8) & (df['coldu'] == 1)][feature].std()
# 		df.loc[(df['pt_d'] == 10) & (df['coldu'] == 1) & (df['coldt'] == 0), feature]= ((df[(df['pt_d'] == 10) & (df['coldu'] == 1) & (df['coldt'] == 0)][feature] - mean8) / std8 * std7 + mean7)
# 	return df

# def adjust_single(df, key, feature):
# 	if key == 'uid':
# 		mean7 = df[df['pt_d'] < 8].drop_duplicates(['uid'])[feature].mean()
# 		std7 = df[df['pt_d'] < 8].drop_duplicates(['uid'])[feature].std()
# 		mean8 = df[(df['coldu'] == 1)].drop_duplicates(['uid'])[feature].mean()
# 		std8 = df[(df['coldu'] == 1)].drop_duplicates(['uid'])[feature].std()
# 		df.loc[(df['pt_d'] == 10) & (df['coldu'] == 1) & (df['coldt'] == 0), feature]= ((df[(df['pt_d'] == 10) & (df['coldu'] == 1) & (df['coldt'] == 0)][feature] - mean8) / std8 * std7 + mean7)
# 	return df


def adjust_single(df, key, feature):
    if key == 'uid':
		mean7 = df[df['pt_d'] < 8].drop_duplicates(['uid'])[feature].mean()
		std7 = df[df['pt_d'] < 8].drop_duplicates(['uid'])[feature].std()
		mean8 = df[(df['pt_d'] >= 8) & (df['coldu'] == 1)].drop_duplicates(['uid'])[feature].mean()
		std8 = df[(df['pt_d'] >= 8) & (df['coldu'] == 1)].drop_duplicates(['uid'])[feature].std()
		df.loc[(df['pt_d'] == 10) & (df['coldu'] == 1) & (df['coldt'] == 0), feature]= ((df[(df['pt_d'] == 10) & (df['coldu'] == 1) & (df['coldt'] == 0)][feature] - mean8) / std8 * std7 + mean7 * 1.1)
		df.loc[(df['pt_d'] == 10) & (df['coldu'] == 1) & (df['coldt'] == 1), feature]= ((df[(df['pt_d'] == 10) & (df['coldu'] == 1) & (df['coldt'] == 1)][feature] - mean8) / std8 * std7 * 0.8 + mean7 * 0.8)
	return df

def group_fea(df,key,target):
    tmp = df.group_by(key,as_index=False)[target].agg({key+'_'+target+'_nunique':'nunique'}).reset_index().drop('index',axis=1)
    return tmp
    
def emb(df,f1,f2):
    emb_size = 8
    tmp = df.group_by(f1,as_index=False)[f2].agg({'{}_{}_list'.format(f1,f2):list})
    sentences = tmp[]
    
