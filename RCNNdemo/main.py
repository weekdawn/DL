# *-* coding:utf-8 *-*
from rcnn import RCNN
from data_utils import data_preprocess
import numpy as np
from keras.callbacks import EarlyStopping

#set parameters:
max_features = 3
max_len = 32
embedding_dims = 32
class_num = 5
batch_size = 32
epochs = 10
origin_filename = "./data/mark_data.xls"

print('Loading data...')
(x_train, x_test), (y_train, y_test) = data_preprocess(origin_filename)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


print('Prepare input for model...')
x_train_left = x_train
x_train_right = x_train.reverse()
x_test_left = x_test
x_test_right = x_test.reverse()
print('x_train_left shape:', np.array(x_train_left).shape)
print('x_train_right shape:', np.array(x_train_right).shape)
print('x_test_left shape:', np.array(x_test_left).shape)
print('x_test_right shape:', np.array(x_test_right).shape)

print('Build model...')
model = RCNN(max_len, max_features, embedding_dims).get_model()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

print('Train...')
early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
model.fit([x_train_left, x_train_right], y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping],
          validation_data=([x_test_left, x_test_right], y_test))

print('Test...')
result = model.predict([x_test_left, x_test_right])