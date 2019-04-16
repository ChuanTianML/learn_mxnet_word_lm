# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os, gzip
import sys
import mxnet as mx
import numpy as np

class Dictionary(object):
    """字典类
    @func add_word(word): 在字典中添加单词word
    """
    def __init__(self):
        self.word2idx = {}      #单词到id
        self.idx2word = []      #id到单词
        self.word_count = []    #统计每个单词在语料中出现的次数，index为单词id

    def add_word(self, word):   #尝试添加一个单词
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.word_count.append(0)
        index = self.word2idx[word]
        self.word_count[index] += 1
        return index

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        """
        @param path: 数据所在目录
        """
        self.dictionary = Dictionary()                  #构造字典实例，准备根据语料构造字典
        self.train = self.tokenize(path + 'train.txt')  #tokenize train/valid/test语料，同时获得字典
        self.valid = self.tokenize(path + 'valid.txt')
        self.test = self.tokenize(path + 'test.txt')

    def tokenize(self, path):
        """构建词表，tokenize语料（转wordid）
        @param path: 语料文件路径
        @return:     转为wordid的语料, 形状为（token数量,）
        @notes：     1.添加了句子结束符'<eos>'
                     2.语料中所有token均被添加到字典
                     3.最后的ids怎么不分行，而是把整个语料文件存进一个长数组？
        """
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0                                  #tokens记录整个文件的token数量
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = np.zeros((tokens,), dtype='int32')    #ids是整个语料文件所有token的wordid
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return mx.nd.array(ids, dtype='int32')

def batchify(data, batch_size):
    """
    @param data:        (Corpus.[train/valid/test]) tokenize后的数据
    @param batch_size:  batch size
    @return:            按batch分好的数据，形状为(batch数量，batch size)
    @notes:             source corpus: [我,爱,你,们,大,家,好,啊,晚上,吃,的,什么,你,是,哪,位,今天,天气,怎么,样,不,告,诉,你]
                        reshape(3,8):   [[我,  爱,  你,  们,  大, 家, 好, 啊],
                                        [晚上, 吃,  的,  什么, 你, 是, 哪, 位],
                                        [今天, 天气, 怎么, 样, 不, 告, 诉, 你]]
                            即reshape((batch_size=3, nbatch=8)，得到形状(batch_size, batch_num*sentence_len)
                            最清晰的数据形状应该是(batch_num, batch_size, sentence_len)，因为这里仅仅保留了2个维度，所以nbatch=batch_num*sentence_len，所以上面的形状不直观

                        T:             [[我, 晚上, 今天],
                                        [爱, 吃,   天气],
                                        [你, 的,   怎么],
                                        [们, 什么, 样]
                                        [大, 你,   不]
                                        [家, 是,   告]
                                        [好, 哪,   诉]
                                        [啊, 位,   你]]
                            得到形状(batch_num*sentence_len, batch_size)

                        iter_next()函数取一个batch的操作是：假设bptt=4，也就是上面每个句子的长度
                            第一次取得到: [[我, 晚上, 今天],
                                          [爱, 吃,   天气],
                                          [你, 的,   怎么],
                                          [们, 什么, 样]]
                            第二次取得到: [[大, 你,   不]
                                          [家, 是,   告]
                                          [好, 哪,   诉]
                                          [啊, 位,   你]]
                            即，在0维度上，一次取一个sentence_len，也就是去了batch_num次
    """
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size            #获取batch的数量，1.从这里的逻辑来看，batch_size单位是token而不是句子？ 2.使用整数除法，尾巴舍弃不要了啊？
    data = data[:nbatch * batch_size]               #两个目的吧，一是转list，二是去除尾巴，即每个batch都是满的
    data = data.reshape((batch_size, nbatch)).T     #转形状，为(batch数量，batch_size)
    return data

class CorpusIter(mx.io.DataIter):
    """数据迭代器
    """
    "An iterator that returns the a batch of sequence each time"
    def __init__(self, source, batch_size, bptt):
        """初始化数据迭代器
        @param source:      (Corpus.[train/valid/test]) tokenize后的数据
        @param batch_size:  batch size
        @param bptt:        句子长度
        """
        super(CorpusIter, self).__init__()
        self.batch_size = batch_size
        self.provide_data = [('data', (bptt, batch_size), np.int32)]    #一个list，只有一个tuple元素，tuple有3个元素
        self.provide_label = [('label', (bptt, batch_size))]            #一个list，只要一个tuple元素，tuple有2个元素
        self._index = 0
        self._bptt = bptt
        self._source = batchify(source, batch_size)                     #数据按batch分好，得到形状为(batch数量，batch size)的数据

    def iter_next(self):
        """mxnet: move to the next batch
        """
        i = self._index                                                         #记录当前取到的位置
        if i+self._bptt > self._source.shape[0] - 1:
            return False
        self._next_data = self._source[i:i+self._bptt]                          #得到形状(bptt, batch_size)
        self._next_label = self._source[i+1:i+1+self._bptt].astype(np.float32)  #得到形状(bptt, batch_size)
        self._index += self._bptt
        return True

    def next(self):
        """mxnet: get next data batch from iterator
        """
        if self.iter_next():                                                    #还有数据可取，则返回数据
            return mx.io.DataBatch(data=self.getdata(), label=self.getlabel())
        else:                                                                   #数据已经取完，则抛出终止迭代错误
            raise StopIteration

    def reset(self):
        self._index = 0
        self._next_data = None
        self._next_label = None

    def getdata(self):
        """mxnet: get data of current batch
        """
        return [self._next_data]

    def getlabel(self):
        """mxnet: get label of current batch
        """
        return [self._next_label]
