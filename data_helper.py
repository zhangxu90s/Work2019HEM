# -*- coding: utf-8 -*-

import re
import pickle
import numpy as np
from gensim import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# 读取数据
def read_data(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = f.readlines()
    data = [re.split('\t', i) for i in data]
    q1 = [i[1] for i in data]
    
    q2 = [i[2] for i in data]
    label = [int(i[3]) for i in data]
    return q1, q2, label


train_q1, train_q2, train_label = read_data('./data/ch_input/BQ_train1.txt')
test_q1, test_q2, test_label = read_data('./data/ch_input/BQ_test1.txt')
dev_q1, dev_q2, dev_label = read_data('./data/ch_input/BQ_dev1.txt')

# 构造训练word2vec的语料库
corpus = train_q1 + train_q2 + test_q1 + test_q2 + dev_q1 + dev_q2
w2v_corpus = [i.split() for i in corpus]
word_set = set(' '.join(corpus).split())

MAX_SEQUENCE_LENGTH = 30  # sequence最大长度为30个词
EMB_DIM = 300  # 词向量为300维

# 训练word2vec模型
w2v_model = models.Word2Vec(w2v_corpus, size=EMB_DIM, window=5, min_count=1, sg=1, workers=4, seed=1234, iter=25)
w2v_model.save('w2v_model.pkl')
#Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类
tokenizer = Tokenizer(num_words=len(word_set))
#要用以训练的文本列表
tokenizer.fit_on_texts(corpus)
#编码转为序列文本
train_q1 = tokenizer.texts_to_sequences(train_q1)
train_q2 = tokenizer.texts_to_sequences(train_q2)

test_q1 = tokenizer.texts_to_sequences(test_q1)
test_q2 = tokenizer.texts_to_sequences(test_q2)

dev_q1 = tokenizer.texts_to_sequences(dev_q1)
dev_q2 = tokenizer.texts_to_sequences(dev_q2)

#转化为长度相同的序列文本
train_pad_q1 = pad_sequences(train_q1, maxlen=MAX_SEQUENCE_LENGTH)
train_pad_q2 = pad_sequences(train_q2, maxlen=MAX_SEQUENCE_LENGTH)

test_pad_q1 = pad_sequences(test_q1, maxlen=MAX_SEQUENCE_LENGTH)
test_pad_q2 = pad_sequences(test_q2, maxlen=MAX_SEQUENCE_LENGTH)

dev_pad_q1 = pad_sequences(dev_q1, maxlen=MAX_SEQUENCE_LENGTH)
dev_pad_q2 = pad_sequences(dev_q2, maxlen=MAX_SEQUENCE_LENGTH)

'''    
#词向量
rows = []
#词表
words = []
#遍历分词后的所有的词
for word_ in word_set:
    #词表长度
    i = len(word_)
    #当分词后的大词表长度>0
    while len(word_) > 0:
        #取词
        word = word_[:i]
        #遍历是否在训练完成后的词向量对应词表中
        if word in w2v_wordmap:
            #将相应的词及其向量对应存储到words和rows中            
            rows.append(w2v_wordmap[word])
            words.append(word)
            word_ = word_[i:]
            i = len(word_)
            continue
        #如果word不在w2v_wordmap，i=i-1,进行下一个词的查找
        else:
            i = i-1
        '''  '''  
        #w2v_wordmap中没有遍历到的词，进行随机向量化
        if i == 0:
            # word OOV
            # https://arxiv.org/pdf/1611.01436.pdf
            rows.append(fill_unk(word_))
            words.append(word_)
            break
        '''
'''        
#词向量-->wvecs[]
wvecs = []
count = 0
for word in word_set:
    count +=1
    wvecs.append(count)

#转置
s = np.vstack(wvecs)
# Gather the distribution hyperparameters
#方差
v = np.var(s,0) 
#平均
m = np.mean(s,0) 
#随机
RS = np.random.RandomState()
#OOV
def fill_unk(unk):
    w2v = {}
    #m-->mean,np.diag(v)-->协方差矩阵
    #np.random.multivariate_normal方法-->用于根据实际情况生成一个多元正态分布矩阵
    #RS.multivariate_normal-->Draw random samples from a multivariate normal distribution.
    #numpy.diag()-->返回一个矩阵的对角线元素，或者创建一个对角阵（ diagonal array.）
    w2v[unk] = int(RS.multivariate_normal(m,np.diag(v)))
    return w2v[unk]
w2v_wordmap = {}
#抽取训练好的词向量中的词以及对应词向量
for word, idx in tokenizer.word_index.items():

    w2v_wordmap[word] = idx
''''''
embedding_matrix = np.zeros([len(tokenizer.word_index) + 1000, EMB_DIM])
#在所有词中查找
for word_ in word_set :
    if word_ in w2v_wordmap:
        idx = w2v_wordmap.get(word_)
        embedding_matrix[idx, :] = w2v_model.wv[word_]
    else:
        tokenizer.word_index.setdefault(word_,fill_unk(word_))
        for word_,idx in tokenizer.word_index.items(): 
            embedding_matrix[idx, :] = w2v_model.wv[word_] 
'''
embedding_matrix = np.zeros([len(tokenizer.word_index) + 1, EMB_DIM])

for word, idx in tokenizer.word_index.items():
    embedding_matrix[idx, :] = w2v_model.wv[word]
    
def save_pickle(fileobj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(fileobj, f)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        fileobj = pickle.load(f)
    return fileobj


model_data = {'train_q1': train_pad_q1, 'train_q2': train_pad_q2, 'train_label': train_label,
              'test_q1': test_pad_q1, 'test_q2': test_pad_q2, 'test_label': test_label,
              'dev_q1': dev_pad_q1, 'dev_q2': dev_pad_q2, 'dev_label': dev_label}

save_pickle(corpus, 'corpus.pkl')
save_pickle(model_data, 'model_data.pkl')
save_pickle(embedding_matrix, 'embedding_matrix.pkl')
save_pickle(tokenizer, 'tokenizer.pkl')
