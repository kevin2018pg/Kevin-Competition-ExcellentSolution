'''
Author: Kevin
Date: 2020-12-09 15:45:35
LastEditTime: 2020-12-10 19:33:36
LastEditors: Please set LastEditors
Description: Training and reasoning of full feature model
FilePath: \HuaweiDigixCTR\code\full.py
'''

# here put the import lib
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from reduce.reduce import reduce, reduce_s
from gensim.models import Word2Vec
import logging
import lightgbm as lgb
from multiprocessing import Pool
import networkx as nx



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
	tmp = df.groupby(f1, as_index=False)[f2].agg({f'{f1}_{f2}_list': list})
	sentences = tmp[f'{f1}_{f2}_list'].values.tolist()
	del tmp[f'{f1}_{f2}_list']
	for i in range(len(sentences)):
		sentences[i] = [str(x) for x in sentences[i]]
	model = Word2Vec(sentences, size=emb_size, window=6, min_count=5, sg=0, hs=0, seed=1, iter=5)
	emb_matrix = []
	for seq in sentences:
		vec = []
		for w in seq:
			if w in model.wv.vocab:
				vec.append(model.wv[w])
		if len(vec) > 0:
			emb_matrix.append(np.mean(vec, axis=0))
		else:
			emb_matrix.append([0] * emb_size)
	emb_matrix = np.array(emb_matrix)
	for i in range(emb_size):
		tmp[f'{f1}_{f2}_emb_{i}'] = emb_matrix[:, i]
	return tmp


def emb2(df,f1,f2):
    emb_size = 8
	tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
	sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
	del tmp['{}_{}_list'.format(f1, f2)]
	for i in range(len(sentences)):
		sentences[i] = [str(x) for x in sentences[i]]
	model = Word2Vec(sentences, size=emb_size, window=6, min_count=5, sg=0, hs=0, seed=1, iter=5)
	emb_matrix = []
	for seq in sentences:
		vec = []
		for w in seq:
			if w in model.wv.vocab:
				vec.append(model.wv[w])
		if len(vec) > 0:
			emb_matrix.append(np.mean(vec, axis=0))
		else:
			emb_matrix.append([0] * emb_size)
	emb_matrix = np.array(emb_matrix)
	for i in range(emb_size):
		tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
	
	word_list = []
	emb_matrix2 = []
	for w in model.wv.vocab:
		word_list.append(w)
		emb_matrix2.append(model.wv[w])
	emb_matrix2 = np.array(emb_matrix2)
	tmp2 = pd.DataFrame()
	tmp2[f2] = np.array(word_list).astype('int')
	for i in range(emb_size):
		tmp2['{}_{}_emb_{}'.format(f2, f1, i)] = emb_matrix2[:, i]
	return tmp, tmp2
	

def emb_adjust(df,f1,f2):
	emb_size = 8
	df = df.fillna(0)
	tmp = df.groupby(f1, as_index=False)[f2].agg({f'{f1}_{f2}_list': list})
	sentences = tmp[f'{f1}_{f2}_list'].values.tolist()
	for i in range(len(sentences)):
		sentences[i] = [str(x) for x in sentences[i]]
	model = Word2Vec(sentences, size=emb_size, window=6, min_count=5, sg=0, hs=0, seed=1, iter=5)

	index_dict = {}
	emb_matrix = []
	for i in tqdm(range(len(sentences))):
		seq = sentences[i]
		vec = []
		for w in seq:
			if w in model.wv.vocab:
				vec.append(model.wv[w])
		if len(vec) > 0:
			emb_matrix.append(np.mean(vec, axis=0))
		else:
			emb_matrix.append([0] * emb_size)
		index_dict[tmp[f1][i]] = i
	emb_matrix = np.array(emb_matrix)
	for i in range(emb_size):
		tmp[f'{f1}_of_{f2}_emb_{i}'] = emb_matrix[:, i]
	
	tmp_f2 = df.groupby(f2, as_index=False)[f1].agg({f'{f2}_{f1}_list':list})
	sentences_f2 = tmp_f2[f'{f2}_{f1}_list'].values.tolist()
	index_dict_f2 = {}
	emb_matrix_f2 = []
	for i in tqdm(range(len(sentences_f2))):
		seq = sentences_f2[i]
		vec = []
		for w in seq:
			vec.append(emb_matrix[index_dict[w]])
		if len(vec) > 0:
			emb_matrix_f2.append(np.mean(vec, axis=0))
		else:
			emb_matrix_f2.append([0] * emb_size)
		index_dict_f2[str(tmp_f2[f2][i])] = i
	emb_matrix_f2 = np.array(emb_matrix_f2)

	emb_matrix_adjust = []
	for seq  in tqdm(sentences):
		vec = []
		for w in seq:
			vec.append(emb_matrix_f2[index_dict_f2[w]])
		if len(vec) > 0:
			emb_matrix_adjust.append(np.mean(vec, axis=0))
		else:
			emb_matrix_adjust.append([0] * emb_size)
	emb_matrix_adjust = np.array(emb_matrix_adjust)
	for i in range(emb_size):
		tmp[f'{f1}_of_{f2}_emb_adjust_{i}'] = emb_matrix_adjust[:, i]

	tmp = tmp.drop('{}_{}_list'.format(f1, f2), axis=1)

	word_list = []
	emb_matrix2 = []
	for w in tqdm(model.wv.vocab):
		word_list.append(w)
		emb_matrix2.append(model.wv[w])
	emb_matrix2 = np.array(emb_matrix2)
	tmp2 = pd.DataFrame()
	tmp2[f2] = np.array(word_list).astype('int')
	for i in range(emb_size):
		tmp2['{}_emb_{}'.format(f2, i)] = emb_matrix2[:, i]
	
	return tmp, tmp2
		


def randomWalk(_g, _corpus_num, _deep_num, _current_word):
	_corpus = []
	for _ in range(_corpus_num):
		sentence = [_current_word]
		current_word = _current_word
		count = 0
		while count<_deep_num:
			count+=1
			_node_list = list(_g[current_word].keys())
			_weight_list = np.array([item['weight'] for item in (_g[current_word].values())])
			_ps = _weight_list / np.sum(_weight_list)
			sel_node = roulette(_node_list, _ps)
			if count % 2 == 0:
				sentence.append(sel_node)
			current_word = sel_node
		_corpus.append(sentence)
	return _corpus

def roulette(_datas, _ps):
	return np.random.choice(_datas, p=_ps)


def build_graph(df,f1,f2):
	G = nx.Graph()
	df_weight = df.groupby([f1,f2],as_index=False)['gender'].agg('weight':'count',).reset_index().drop('index',axis=1)
	df_weight[f1+'_word'] = df_weight[f1].astype(str)+'_'+f1
	df_weight[f2+'_word'] = df_weight[f2].astype(str)+'_'+f2
	df_weight df_weight.drop(f1,axis=1).drop(f2,axis=1)
	for i in tqdm(range(len(df_weight))):
		G.add_edge(df_weight[f1 + '_word'][i], df_weight[f2 + '_word'][i], weight=df_weight['weight'][i])
	return G,df_weight


def deep_walk(G,df_weight,f1,f2):
	num = 5
	deep_num = 20
	f2_set = set(df_weight[f2 + '_word'])
	sentences = []
	for word in tqdm(f2_set):
		corpus = randomWalk(G, num, deep_num, word)
		sentences += corpus
	return sentences

	
	

			

	
	
		



    

    
