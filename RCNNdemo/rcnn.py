# *-* coding:utf-8 *-*
from keras import Input, Model
from keras import backend as K
from keras.layers import Embedding, Dense, SimpleRNN, Lambda, Concatenate, Conv1D
from keras.layers import GlobalAveragePooling1D

class RCNN(object):
    def __init__(self, max_len, max_feature, embedding_dims, class_num=1):
        self.max_feature = max_feature
        self.max_len = max_len
        self.embedding_dims = embedding_dims
        self.class_num = class_num

    def get_model(self):
        # 输入层
        input_left = Input((self.max_len,))
        input_right = Input((self.max_len,))
        # 第一层
        embedder = Embedding(self.max_feature, self.embedding_dims, input_length=self.max_len)
        embedding_left = embedder(input_left)
        embedding_right = embedder(input_right)
        # 第二层
        rnn_left = SimpleRNN(128, return_sequences=True)(embedding_left)
        rnn_right = SimpleRNN(128, return_sequences=True, go_backwards=True)(embedding_right)
        rnn_right = Lambda(lambda x: K.reverse(x, axes=1))(rnn_right)
        rnn = Concatenate(axis=2)([rnn_left, rnn_right])
        # 第三层
        conv = Conv1D(64, kernel_size=1, activation="tanh")(rnn)
        pool = GlobalAveragePooling1D()(conv)
        # 第四层（输出层）
        outputs = Dense(self.class_num, activation="softmax")(pool)
        model = Model(inputs=[input_left, input_right], outputs=outputs)

        return model
