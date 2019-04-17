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

import numpy as np
import mxnet as mx, math
import argparse, math
import logging
from data import Corpus, CorpusIter
from model import rnn, softmax_ce_loss
from module import *
from mxnet.model import BatchEndParam

parser = argparse.ArgumentParser(description='Sherlock Holmes LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/sherlockholmes.',           #数据所在目录，下面有[train/valid/test].txt
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=650,          #embedding size 
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=650,            #hidden units num per layer
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,           #层数
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1.0,            #初始学习率
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.2,          #梯度裁剪
                    help='gradient clipping by global norm')
parser.add_argument('--epochs', type=int, default=40,           #epoch数
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32,       #batch size
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.5,               #dropout
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',                      #word embedding 和 softmax 权重共享
                    help='tie the word embedding and softmax weights')
parser.add_argument('--bptt', type=int, default=35,                     #序列长度，为什么起名bptt
                    help='sequence length')
parser.add_argument('--log-interval', type=int, default=200,    #在训练过程中是否汇报？
                    help='report interval')
parser.add_argument('--seed', type=int, default=3,              #随机种子，用来shuf数据？
                    help='random seed')
args = parser.parse_args()

best_loss = 9999

def evaluate(valid_module, data_iter, epoch, mode, bptt, batch_size):
    total_loss = 0.0
    nbatch = 0
    for batch in data_iter:
        valid_module.forward(batch, is_train=False)
        outputs = valid_module.get_loss()
        total_loss += mx.nd.sum(outputs[0]).asscalar()
        nbatch += 1
    data_iter.reset()
    loss = total_loss / bptt / batch_size / nbatch
    logging.info('Iter[%d] %s loss:\t%.7f, Perplexity: %.7f' % \
                 (epoch, mode, loss, math.exp(loss)))
    return loss

if __name__ == '__main__':
    # args
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)   #设置日志格式
    args = parser.parse_args()                              #解析参数
    logging.info(args)
    ctx = mx.gpu()                                          #设置使用gpu还是设置gpu数量
    batch_size = args.batch_size                            #batch size
    bptt = args.bptt                                        #句子长度
    mx.random.seed(args.seed)                               #设置随机种子，shuf数据用？

    # data
    corpus = Corpus(args.data)                              #根据args.data目录下的[train/valid/test].txt 3个文件，构建词表，tokenize语料（转wordid）
    ntokens = len(corpus.dictionary)                        #从语料获得词典大小
    train_data = CorpusIter(corpus.train, batch_size, bptt) #train/valid/test 数据迭代器，初始化仅完成数据按batch切分
    valid_data = CorpusIter(corpus.valid, batch_size, bptt)
    test_data = CorpusIter(corpus.test, batch_size, bptt)

    # model
    pred, states, state_names = rnn(bptt, ntokens, args.emsize, args.nhid,                      #构建rnn网络，返回pred的形状(sbtt*batch_size, vocab_size)，states形状(num_layers, batch_size, length, hidden_size)，state_names形状(num_layers, 2)
                                    args.nlayers, args.dropout, batch_size, args.tied)
    """
    pred:       symbol          (bptt*batch_size, vocab_size) 
    states:     list of symbol  length = num_layers
    states[0]:  symbol          (1, batch_size, hiddle_size)
    """
    loss = softmax_ce_loss(pred)                                                                #得到形状(bptt*batch_size,)

    # module
    module = CustomStatefulModule(loss, states, state_names=state_names, context=ctx)           #构造module，干什么的？
    module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)     #这是干啥的？网上说是分配显存
    module.init_params(initializer=mx.init.Xavier())                                            #初始化参数，mx.init.Xavier()还需学习，默认是均匀分布初始化
    optimizer = mx.optimizer.create('sgd', learning_rate=args.lr, rescale_grad=1.0/batch_size)  #设置优化器，文档中仅仅显示地给出了name，其他的参数没说明，这里很重要
    module.init_optimizer(optimizer=optimizer)                                                  #给module设置优化器

    # metric
    speedometer = mx.callback.Speedometer(batch_size, args.log_interval)                        #定期告知训练速度和验证结果？

    # train
    logging.info("Training started ... ")
    for epoch in range(args.epochs):                                                            #epoch数
        # train
        total_loss = 0.0
        nbatch = 0
        for batch in train_data:                                                                #batch: mx.io.DataBatch(data=self.getdata(), label=self.getlabel())
            module.forward(batch)                                                               #前向传播
            module.backward()                                                                   #计算梯度
            module.update(max_norm=args.clip * bptt * batch_size)                               #更新参数
            # update metric
            outputs = module.get_loss()                                                         #形状应该是(1,)？
            total_loss += mx.nd.sum(outputs[0]).asscalar()
            speedometer_param = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                              eval_metric=None, locals=locals())
            speedometer(speedometer_param)
            if nbatch % args.log_interval == 0 and nbatch > 0:
                cur_loss = total_loss / bptt / batch_size / args.log_interval                   #total_loss除这么多东西？网络中的loss不是已经在bptt*batch_size上做了平均了吗？
                logging.info('Iter[%d] Batch [%d]\tLoss:  %.7f,\tPerplexity:\t%.7f' % \
                             (epoch, nbatch, cur_loss, math.exp(cur_loss)))
                total_loss = 0.0
            nbatch += 1
        # validation
        valid_loss = evaluate(module, valid_data, epoch, 'Valid', bptt, batch_size)             #进行验证
        if valid_loss < best_loss:
            best_loss = valid_loss
            # test
            test_loss = evaluate(module, test_data, epoch, 'Test', bptt, batch_size)            #如果出现更好的结果，则进行一次测试
        else:                                                                                   #如果没有出现更好的结果，降低学习率
            optimizer.lr *= 0.25
        train_data.reset()                                                                      #一个epoch训练结束之后，重置训练数据实例
    logging.info("Training completed. ")
