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
import logging

class CustomStatefulModule():
    """CustomStatefulModule is a module that takes a custom loss symbol and state symbols.
    The custom loss is typically composed by `mx.sym.make_loss` or `mx.sym.MakeLoss`.
    The states listed in `state_names` will be carried between iterations.

    Parameters
    ----------
    loss : Symbol
        The custom loss symbol
    states: list of Symbol
        The symbols of next states
    state_names : list of str
        states are similar to data and label, but not provided by data iterator.
        Instead they are initialized to `initial_states` and can be carried between iterations.
    data_names : list of str
        Defaults to `('data')` for a typical model used in image classification.
    label_names : list of str
        Defaults to `('softmax_label')` for a typical model used in image
        classification.
    logger : Logger
        Defaults to `logging`.
    context : Context or list of Context
        Defaults to ``mx.cpu()``.
    initial_states: float or list of NDArray
        Defaults to 0.0.
    """
    def __init__(self, loss, states, state_names, data_names=('data',), label_names=('label',),
                 context=mx.cpu(), initial_states=0.0, **kwargs):
        """
        @param loss:            自定义损失，形状为一个值
        @param states:          所有层所有时间步的next_states，形状为(num_layers, batch_size, length, hidden_size)
        @param data_names   
        @param label_names
        @param context:     
        @param initial_states:
        """
        if isinstance(states, mx.symbol.Symbol):                        #states放进数组
            states = [states]
        self._net = mx.sym.Group(states + [loss])                       #将[states, loss]构成一个symbol，这个symbol决定了module的输出，决定了module.get_outputs()的返回值
        self._next_states = initial_states                              #形状是(num_layers*batch_size, 2, hidden_size)？
        self._module = mx.module.Module(self._net,                      #symbol=self._net, 要获得这个symbol的前向传播结果
                                        data_names=data_names, 
                                        label_names=label_names,  
                                        context=context, 
                                        state_names=state_names, 
                                        **kwargs)
        #Module is a basic module that wrap a Symbol. It is functionally the same as the FeedForward model, except under the module API.

    def backward(self, out_grads=None):
        """Backward computation. 梯度回传
        @param out_grads: (NDArray or list of NDArray, optional), Gradient on the outputs to be propagated back. This parameter is only needed when bind is called on outputs that are not a loss function.
        """
        self._module.backward(out_grads=out_grads)

    def init_params(self, initializer=mx.init.Uniform(0.01), **kwargs):
        """Initializes the parameters and auxiliary states.
        """
        self._module.init_params(initializer=initializer, **kwargs)

    def init_optimizer(self, **kwargs):
        """Installs and initializes optimizers, as well as initialize kvstore for
           distributed training.
        """
        self._module.init_optimizer(**kwargs)

    def bind(self, data_shapes, **kwargs):
        """Binds the symbols to construct executors. This is necessary before one
        can perform computation with the module.
        """
        self._module.bind(data_shapes, **kwargs)

    def forward(self, data_batch, is_train=None, carry_state=True):
        """Forward computation. States from previous forward computation are carried
        to the current iteration if `carry_state` is set to `True`.
        之前前向传播的states被带到这次迭代？ states是句子最后一个时间步的states，形状是(num_layers*batch_size, 2, hidden_size)，带到这次迭代做什么用？
        @param data_batch: 一个batch的数据mx.io.DataBatch(data=self.getdata(), label=self.getlabel()), data和label的形状都是(1, bptt, batch_size)
        @param is_train:
        @param carry_state:
        """
        # propagate states from the previous iteration
        if carry_state:                                                 #带过来上一次迭代（上一个batch）的states
            if isinstance(self._next_states, (int, float)):             #统一设成一个值
                self._module.set_states(value=self._next_states)        #mxnet: Sets value for states. Only one of the states & value can be specified.
            else:                                                       #也是一个具有多维形状的states，用此设置这次的初始states？
                self._module.set_states(states=self._next_states)
        self._module.forward(data_batch, is_train=is_train)             #前向传播
        outputs = self._module.get_outputs(merge_multi_context=False)   #mxnet: Gets outputs of the previous forward computation. 形状是()？
        """outputs里面到底是什么东西？
                从下面一行看，除了倒数第一个值，其他位置存的是states，states的形状是？
                从另一处引用get_outputs()看，最后一个元素存的是loss，loss的形状是？根据对loss的处理，这里的loss是没有对btpp,batch_size取平均的
        """
        self._next_states = outputs[:-1]

    def update(self, max_norm=None):
        """Updates parameters according to the installed optimizer and the gradients computed
        in the previous forward-backward batch. Gradients are clipped by their global norm
        if `max_norm` is set.

        Parameters
        ----------
        max_norm: float, optional
            If set, clip values of all gradients the ratio of the sum of their norms.
        """
        if max_norm is not None:
            self._clip_by_global_norm(max_norm)
        self._module.update()

    def _clip_by_global_norm(self, max_norm):
        """Clips gradient norm.

        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.
        The method is first used in
         `[ICML2013] On the difficulty of training recurrent neural networks`

        Parameters
        ----------
        max_norm : float or int
            The maximum clipping threshold of the gradient norm.

        Returns
        -------
        norm_val : float
            The computed norm of the gradients.
        """
        assert self._module.binded and self._module.params_initialized \
               and self._module.optimizer_initialized
        grad_array = []
        for grad in self._module._exec_group.grad_arrays:
            grad_array += grad
        return mx.gluon.utils.clip_global_norm(grad_array, max_norm)

    def get_loss(self):
        """Gets the output loss of the previous forward computation.
        """
        return self._module.get_outputs(merge_multi_context=False)[-1]
