from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence

from TextAttBiRNN import TextAttBiRNN, load_imdb_data

'''
更多 中文长文本分类、短句子分类、多标签分类、两句子相似度 的模型,请参考：
https://github.com/yongzhuo/Keras-TextClassification
'''

max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 10

print('Loading data...')
(x_train, y_train), (x_test, y_test) = load_imdb_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build TextAttBiRNN model for text classification ...')
model = TextAttBiRNN(maxlen, max_features, embedding_dims)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
# model_ckpt = ModelCheckpoint(filepath= './ckpt/')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping],
          validation_data=(x_test, y_test))

print('Test...')
result = model.predict(x_test)

