# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import Model
from keras.layers import Layer, Embedding, Dense,  Bidirectional, LSTM
from keras import backend as K
from keras import initializers, regularizers, constraints

''''
用于事件分类的注意力机制： 
'''

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name),
                                     shape=(input_shape[1],),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


class TextAttBiRNN(Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num=1,
                 last_activation='sigmoid'):
        super(TextAttBiRNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims    # 128
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.bi_rnn = Bidirectional(LSTM(self.embedding_dims, return_sequences=True))  # LSTM or GRU
        self.attention = Attention(self.maxlen)
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextAttBiRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextAttBiRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        embedding = self.embedding(inputs)
        x = self.bi_rnn(embedding)
        x = self.attention(x)
        output = self.classifier(x)
        return output






# ========= load imdb data  =========

from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import _remove_long_seq
import numpy as np
import json
import warnings

def load_imdb_data(path = r'D:/projects/dataset/imdb.npz', num_words=None, skip_top=0,
              maxlen=None, seed=113,
              start_char=1, oov_char=2, index_from=3, **kwargs):
    """Loads the IMDB dataset.

    # Arguments
        path: imdb.npz 的路径.
        num_words: max number of words to include.
        skip_top: skip the top N most frequently occurring words
            (which may not be informative).
        maxlen: sequences longer than this will be filtered out.
        seed: random seed for sample shuffling.
        start_char: The start of a sequence will be marked with this character.
            Set to 1 because 0 is usually the padding character.
        oov_char: words that were cut out because of the `num_words`
            or `skip_top` limit will be replaced with this character.
        index_from: index actual words with this index and higher.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case `maxlen` is so low
            that no input sequence could be kept.

    """
    # Legacy support
    if 'nb_words' in kwargs:
        warnings.warn('The `nb_words` argument in `load_data` '
                      'has been renamed `num_words`.')
        num_words = kwargs.pop('nb_words')
    if kwargs:
        raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))


    with np.load(path,allow_pickle=True) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                             'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x]
              for x in xs]
    else:
        xs = [[w for w in x if skip_top <= w < num_words]
              for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)

# =============
from keras.callbacks import Callback


#! -*- coding: utf-8 -*-


import json
from tqdm import tqdm
import os
import numpy as np
import pandas as pd


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adam
'''
ee-2019-baseline：面向金融领域的事件主体抽取的一个baseline
（ccks2019，https://biendata.com/competition/ccks_2019_4/ ），
模型 ：用BiLSTM ＋ 指针结构标注实体。

实际上这个比赛就是阅读理解竞赛SQUAD 1.0的简化版：
它要输入“一段文本”+“事件类型”，输出文本中的实体，
如果将“事件类型”看成问题，将“一段文本”看作是篇章，那么它就跟squad 1.0的格式一模一样了，
任何squad的模型都可以简化后用到这个问题上。

用法 ：python ee.py即可。gtx 1060上30秒训练一个epoch（包括验证时间）。
结果： 取决于验证集划分的不同，线下验证集的acc大概是0.76+左右。
亲测线上提交可以达到0.83+，如果你的解码规则写得好，估计可以到0.84+。
详细：https://github.com/Xiefan-Guo/CCKS2019_subject_extraction
'''

mode = 0
min_count = 2
char_size = 128
maxlen = 256

train_data = '../data/event_type_entity_extract_train.csv'
char_id = '../temp_data/all_chars_me.json'
class_id = '../temp_data/classes.json'
random_order_json = '../temp_data/random_order_train.json'
tes_result = '../temp_data/result.txt'

# 模型权重保存的hdf5文件路径
model_weights = '../ckpt/best_model.weights'

# 读取数据，排除“其他”类型 data = [ [id, texts, class_label, subject] ]

D = pd.read_csv(train_data, encoding='utf-8', header=None)
D = D[D[2] != u'其他']
D = D[D[1].str.len() <= maxlen]

# D[条件] 读入D按条件进行过滤

if not os.path.exists(class_id):
    #  D[2].unique()提取所有的subject：id2class：  {id：谓语}
    id2class = dict(enumerate(D[2].unique()))
    print('id2class:  ', id2class)

    # class2id： {谓语：id}
    class2id = {j:i for i,j in id2class.items()}
    print('class2id :  ', class2id )

    # 生成一个谓语的字典，存入classes.json中
    json.dump([id2class, class2id], open(class_id, 'w', encoding='utf-8'))
else:
    id2class, class2id = json.load(open(class_id, 'r', encoding='utf-8'))

print('D[2].unique():  ',D[2].unique())

train_data = []
for t,c,n in zip(D[1], D[2], D[3]):
    """
    t:整个事件 texts
    c:谓语  class
    n:主语 noun-名词
    """
    start = t.find(n)
    if start != -1:
        """
        在事件中可以找到主语就将其append到train_data
        """
        train_data.append((t, c, n))

if not os.path.exists(char_id):
    chars = {}
    for d in tqdm(iter(train_data)):
        # print(d)
        for c in d[0]:
            chars[c] = chars.get(c, 0) + 1
    # print(chars)
    """
    获取训练数据集中所有事件中出现的所有“字”（字符）
    最终字典形式为: key='字符'，value=该字符出现个数
    """
    chars = {i:j for i,j in chars.items() if j >= min_count}
    id2char = {i+2:j for i,j in enumerate(chars)} # 0: mask, 1: padding
    char2id = {j:i for i,j in id2char.items()}
    """
    为所有的字符建立索引，下标从2开始
    """
    json.dump([id2char, char2id], open(char_id, 'w', encoding='utf-8'))
else:
    id2char, char2id = json.load(open(char_id, 'r', encoding='utf-8'))


if not os.path.exists(random_order_json):
    random_order = list(range(len(train_data)))
    np.random.shuffle(random_order)
    json.dump(random_order, open(random_order_json, 'w', encoding='utf-8'))
else:
    random_order = json.load(open(random_order_json, 'r', encoding='utf-8'))

""" 把数据按照9:1分为训练集和验证集 :"""
dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == mode]
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]


D = pd.read_csv('../data/event_type_entity_extract_eval.csv', encoding='utf-8', header=None)
test_data = []
for id,t,c in zip(D[0], D[1], D[2]):
    test_data.append((id, t, c))
"""测试数据test_data 一个list :内部每个元组为：（编号，事件，主体）"""


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

class data_generator:
    def __init__(self, data, batch_size=16):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X, C, S1, S2 = [], [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0]
                x = [char2id.get(c, 1) for c in text]
                c = class2id[d[1]]
                s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                start = text.find(d[2])
                end = start + len(d[2]) - 1
                s1[start] = 1
                s2[end] = 1
                X.append(x)
                C.append([c])
                S1.append(s1)
                S2.append(s2)
                if len(X) == self.batch_size or i == idxs[-1]:
                    X = seq_padding(X)
                    C = seq_padding(C)
                    S1 = seq_padding(S1)
                    S2 = seq_padding(S2)
                    yield [X, C, S1, S2], None
                    X, C, S1, S2 = [], [], [], []


class Attention(Layer):
    """多头注意力机制
    """
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head

    def build(self, input_shape):
        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        self.q_kernel = self.add_weight(name='q_kernel',
                                        shape=(q_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.k_kernel = self.add_weight(name='k_kernel',
                                        shape=(k_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.v_kernel = self.add_weight(name='w_kernel',
                                        shape=(v_in_dim, self.out_dim),
                                        initializer='glorot_normal')
    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                # ndim以整数形式返回张量中的轴数。
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10
    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变化
        qw = K.dot(q, self.q_kernel)
        kw = K.dot(k, self.k_kernel)
        vw = K.dot(v, self.v_kernel)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.size_per_head**0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


x_in = Input(shape=(None,)) # 待识别句子输入
c_in = Input(shape=(1,)) # 事件类型
s1_in = Input(shape=(None,)) # 实体左边界（标签）
s2_in = Input(shape=(None,)) # 实体右边界（标签）
"""
Input用来实例化一个keras张量
Input(shape=None,batch_shape=None,name=None,dtype=K.floatx(),sparse=False,tensor=None)
"""

x, c, s1, s2 = x_in, c_in, s1_in, s2_in
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
# Lamda : 将任意表达式封装为 Layer 对象。

x = Embedding(len(id2char)+2, char_size)(x)
# Embedding 将正整数（索引值）转换为固定尺寸的稠密向量。 例如： [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
# 该层只能用作模型中的第一层。
c = Embedding(len(class2id), char_size)(c)
c = Lambda(lambda x: x[0] * 0 + x[1])([x, c])
x = Add()([x, c])
x = Dropout(0.2)(x)
# Dropout 将 Dropout 应用于输入。
x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

x = Bidirectional(LSTM(char_size // 2, return_sequences=True))(x)
# x = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x)

# CuDNNLSTM 由 CuDNN 支持的快速 LSTM 实现。只能以 TensorFlow 后端运行在 GPU 上
# Bidirectional RNN 的双向封装器，对序列进行前向和后向计算。
x = Lambda(lambda x: x[0] * x[1])([x, x_mask])
x = Bidirectional(LSTM(char_size // 2, return_sequences=True))(x)
# x = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x)
x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

xo = x
x = Attention(8, 16)([x, x, x, x_mask, x_mask])
x = Lambda(lambda x: x[0] + x[1])([xo, x])

x = Concatenate()([x, c])
# Concatenate Concatenate 层的函数式接口。

x1 = Dense(char_size, use_bias=False, activation='tanh')(x)
# Dense 全连接层。
ps1 = Dense(1, use_bias=False)(x1)
ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])

x2 = Dense(char_size, use_bias=False, activation='tanh')(x)
ps2 = Dense(1, use_bias=False)(x2)
ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

model = Model([x_in, c_in], [ps1, ps2])

train_model = Model([x_in, c_in, s1_in, s2_in], [ps1, ps2])

loss1 = K.mean(K.categorical_crossentropy(s1_in, ps1, from_logits=True))
# categorical_crossentropy输出张量与目标张量之间的分类交叉熵。
# mean 张量在某一指定轴的均值。
loss2 = K.mean(K.categorical_crossentropy(s2_in, ps2, from_logits=True))
loss = loss1 + loss2

train_model.add_loss(loss)
# 指定自定义的损失函数，通过调用 self.add_loss(loss_tensor)
train_model.compile(optimizer=Adam(1e-3))
# compile用于配置训练模型。 optimizer: 字符串（优化器名）或者优化器实例。
train_model.summary()
# model.summary() 打印出模型概述信息。

def extract_entity(text_in, c_in):
    """解码函数，应自行添加更多规则，保证解码出来的是一个公司名
    """
    if c_in not in class2id:
        return 'NaN'
    _x = [char2id.get(c, 1) for c in text_in]
    _x = np.array([_x])
    _c = np.array([[class2id[c_in]]])
    _ps1, _ps2 = model.predict([_x, _c])     # 为输入样本生成输出预测。
    start = _ps1[0].argmax()
    # 返回指定轴的最大值的索引。keras.backend.argmax(x, axis=-1) x: 张量或变量。 axis: 执行归约操作的轴。
    end = _ps2[0][start:].argmax() + start
    return text_in[start: end+1]

class Evaluate(Callback):
    def __init__(self):
        self.ACC = []
        self.best = 0.
    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            train_model.save_weights('best_model.weights')
            # model.save_weights(filepath) 将模型权重存储为 HDF5 文件。
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))
    def evaluate(self):
        A = 1e-10
        for d in tqdm(iter(dev_data)):
            R = extract_entity(d[0], d[1])
            if R == d[2]:
                A += 1
        return A / len(dev_data)



def test(test_data):
    """注意官方页面写着是以\t分割，实际上却是以逗号分割
    """
    F = open(tes_result, 'wb+', encoding='utf-8')
    for d in tqdm(iter(test_data)):
        s = u'"%s","%s"\n' % (d[0], extract_entity(d[1].replace('\t', ''), d[2]))
        s = s.encode('utf-8')
        F.write(s)
    F.close()


evaluator = Evaluate()
train_D = data_generator(train_data)

# filepath="weights.best.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

train_model.fit_generator(train_D.__iter__(),
                          steps_per_epoch=len(train_D),
                          epochs=1,   # 120
                          callbacks=[evaluator],
                          verbose=0)

if __name__ == '__main__':
    test(test_data)

