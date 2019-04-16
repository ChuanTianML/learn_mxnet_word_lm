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

import mxnet as mx

def rnn(bptt, vocab_size, num_embed, nhid,
        num_layers, dropout, batch_size, tied):
    """
    @param bptt:        句子长度
    @param vocab_size:  词表尺寸
    @param num_embed：  embedding size
    @param nhid：       hidden size 
    @param num_layers   层数
    @param dropout      dropout
    @param batch_size   batch_size
    @param tied         word embedding 和 softmax 是否贡献参数
    """
    # encoder
    data = mx.sym.Variable('data')                                              #名为data的变量，是进行lookup输入的数据？
    weight = mx.sym.var("encoder_weight", init=mx.init.Uniform(0.1))            #是word embedding数据？ 初始化是使用均匀初始化，范围(-0.1, 0.1)。 ps: var和Variable的功能应该是一样的
    embed = mx.sym.Embedding(data=data, weight=weight, input_dim=vocab_size,    #构建word embedding，输出形状为（batch_size, length, embedding_size）未确定？
                             output_dim=num_embed, name='embed')

    # stacked rnn layers
    states = []                                                                                 #保存了每层输出的states，最终形状(num_layers, batch_size, length, hidden_size)
    state_names = []                                                                            #保存了每层输出的states的名字，最终形状(num_layers, 2)
    outputs = mx.sym.Dropout(embed, p=dropout)                                                  #对embedding进行dropout，outputs变量名循环使用，形状为（batch_size, length, embedding_size）未确定？即一整层的输入
    for i in range(num_layers):
        prefix = 'lstm_l%d_' % i                                                                #每层名字的前缀
        cell = mx.rnn.FusedRNNCell(num_hidden=nhid, prefix=prefix, get_next_state=True,         #关于FusedRNNCell，将rnn层跨时间步融合为一个内核，什么？
                                   forget_bias=0.0, dropout=dropout)                            #1.遗忘门的bias被设为了0，为什么？ 2.get_next_state表示是否返回可以用于下一个时间步输入的states
        state_shape = (1, batch_size, nhid)
        begin_cell_state_name = prefix + 'cell'
        begin_hidden_state_name = prefix + 'hidden'
        begin_cell_state = mx.sym.var(begin_cell_state_name, shape=state_shape)                 #应该是第一个时间步的cell输入，疑惑：普通rnn是没有cell输入的吧
        begin_hidden_state = mx.sym.var(begin_hidden_state_name, shape=state_shape)             #应该是第一个时间步的hidden输入
        state_names += [begin_cell_state_name, begin_hidden_state_name]
        outputs, next_states = cell.unroll( bptt,                                               #length=bptt 要展开的时间步数量
                                            inputs=outputs,                                     #(Symbol, list of Symbol, or None), If inputs is a single Symbol (usually the output of Embedding symbol), it should have shape (batch_size, length, ...)
                                            begin_state=[begin_cell_state, begin_hidden_state], #(nested list of Symbol, default None) 
                                            merge_outputs=True,                                 #If False, return outputs as a list of Symbols. If True, concatenate output across time steps and return a single symbol with shape (batch_size, length, ...) 
                                            layout='TNC')                                       #layout of input symbol. Only used if inputs is a single Symbol
        # outputs, next_states都是一整层的输出（即合并了时间步），outputs形状是（batch_size, length, hidden_size），next_states形状是？也是（batch_size, length, hidden_size）？
        outputs = mx.sym.Dropout(outputs, p=dropout)                                            #对输出进行dropout
        states += next_states

    # decoder
    pred = mx.sym.Reshape(outputs, shape=(-1, nhid))                                            #形状转换：(batch_size, length, hidden_size) -> (batch_size*length, hidden_size)
    if tied:                                                                                    #如果softmax和word embedding共享权重
        assert(nhid == num_embed), \
               "the number of hidden units and the embedding size must batch for weight tying"
        pred = mx.sym.FullyConnected(data=pred, weight=weight,
                                     num_hidden=vocab_size, name='pred')
    else:                                                                                       #softmax使用单独的权重
        pred = mx.sym.FullyConnected(data=pred, num_hidden=vocab_size, name='pred')
    pred = mx.sym.Reshape(pred, shape=(-1, vocab_size))                                         #形状转换：(batch_size*length, vocab_size) -> (batch_size*length, vocab_size) 感觉不需要转换形状啊
    return pred, [mx.sym.stop_gradient(s) for s in states], state_names                         #返回pred，阻止了梯度回传的states，state_name

def softmax_ce_loss(pred):
    """构建计算softmax交叉熵损失的网络结构
    @param pred: 模型预测结果，形状为(batch_size*length, vocab_size)
    """
    # softmax cross-entropy loss
    label = mx.sym.Variable('label')                            #输入的label变量
    label = mx.sym.Reshape(label, shape=(-1,))                  #把输入的label拉平成一维，得到形状(batch_size*length,)
    logits = mx.sym.log_softmax(pred, axis=-1)                  #计算logits（先softmax后取log），得到形状(batch_size*length, vocab_size)
    loss = -mx.sym.pick(logits, label, axis=-1, keepdims=True)  #将每个单词的-logits取出来，得到形状(batch_size*length,)
    loss = mx.sym.mean(loss, axis=0, exclude=True)              #计算平均值，得到一个值
    return mx.sym.make_loss(loss, name='nll')                   #用来自定义损失函数巴拉巴拉，还不是很清楚
