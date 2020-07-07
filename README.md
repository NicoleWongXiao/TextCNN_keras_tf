
task1:  使用 imdb.npz数据集+ TextCNN_Classify_keras进行文本分类；在terminal中输入：

cd imdb_TextAttBiRNN
python TextCNN_Classify_keras.py 

由于数据集老是连接超时/下载失败，
下载链接： https://s3.amazonaws.com/text-datasets/imdb.npz  (16.6M大小)
可以使用迅雷先把数据集下载到本地的  D:/projects/dataset/imdb.npz；
再改写一下 imdb.py 的load_data函数到TextAttBiRNN.py 的load_imdb_data 中(注意修改path) ； 
再使用TextAttBiRNN 进行分类；


# =============
task2: 在TensorFlow中实现CNN进行文本分类,在terminal中输入：
python train.py
python evaluation.py

参考网站：https://github.com/dennybritz/cnn-text-classification-tf

完整笔记见http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

数据集下载：http://www.cs.cornell.edu/people/pabo/movie-review-data/的电影评论数据

了解用于NLP的卷积神经网络 http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/

英文和中文的区别就是分词的过程，中文一般使用jieba,或者word2vec(gensim库） 

这里我们加入了三个滤波器区域大小：2,3和4，每个滤波器有2个滤波器。每个滤波器对句子矩阵执行卷积并生成（可变长度）特征映射。然后在每个地图上执行1-max池，即记录来自每个特征地图的最大数目。因此，从所有六个地图生成单变量特征向量，并且这六个特征被连接以形成倒数第二层的特征向量。最后的softmax层接收这个特征向量作为输入，并用它来分类句子; 这里我们假设二进制分类，因此描述了两种可能的输出状态。资料来源：Zhang，Y.，＆Wallace，B。（2015）。

