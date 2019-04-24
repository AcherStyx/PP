from six.moves import cPickle as pickle
import numpy as np 
import os
import platform

import tensorflow as tf
import random
import time

BATCH_SIZE=10
TRAINING_STEPS=50000
DATASET_FOLDER='./dataset/'
#频率
CHECK_FREQUNCY=100
ACCURACY_TEST_DATA_UPDATE_FREQUENCY=5
ACCURACY_TEST_BATCHSIZE=400
#学习率设置
LEARNING_RATE_BASE=0.001
LEARNING_RATE_DECAY_RATE=0.99
LEARNING_RATE_DECAY_STEP=1000
#正则化比率
REGULARIZATION_RATE=0.005

IMAGE_SIZE=[32,32,3]
#层1 卷积层
LAYER1_FILTER_SIZE=[5,5,3,8]
LAYER1_BIASE_SIZE=[LAYER1_FILTER_SIZE[3]]
LAYER1_STRIDES=[1,1,1,1]
#层2 池化层
LAYER2_FILTER_SIZE=[1,2,2,1]
LAYER2_STRIDES=[1,2,2,1]
#层3 卷积层
LAYER3_FILTER_SIZE=[5,5,8,16]
LAYER3_BIASE_SIZE=[LAYER3_FILTER_SIZE[3]]
LAYER3_STRIDES=[1,1,1,1]
#层4 池化层
LAYER4_FILTER_SIZE=[1,2,2,1]
LAYER4_STRIDES=[1,2,2,1]
#层1-? 全连接层
Layer=(1024,200,100,10) 
#总计训练次数
global_step=tf.Variable(0,trainable=False)

#数据集读取
class CIFAR10:
  '''摘自：https://www.cnblogs.com/jimobuwu/p/9161531.html
  自己稍作封装'''
  Train={}
  Train_Size=0
  Test={}
  Test_Size=0
  def __load_pickle(self,f):
      version = platform.python_version_tuple() # 取python版本号
      if version[0] == '2':
          return  pickle.load(f) # pickle.load, 反序列化为python的数据类型
      elif version[0] == '3':
          return  pickle.load(f, encoding='latin1')
      raise ValueError("invalid python version: {}".format(version))
  def __load_CIFAR_batch(self,filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
      datadict = self.__load_pickle(f)   # dict类型
      X = datadict['data']        # X, ndarray, 像素值
      Y = datadict['labels']      # Y, list, 标签, 分类
      # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
      # transpose，转置
      # astype，复制，同时指定类型
      X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
      Y = np.array(Y)
      return X, Y
  def nextbatch(self,batch_size,type='Train'):
    batch_image=[]
    batch_label=[]
    if type=='Train':
      for i in range(batch_size):
        temp=[]
        randnum=random.randint(0,self.Train_Size-1)
        temp.append(self.Train['image'][randnum])
        batch_image.append(temp)
        temp=[0,0,0,0,0,0,0,0,0,0]
        temp[int(self.Train['label'][randnum])]=1
        batch_label.append(temp)
      batch_image=np.reshape(batch_image,[batch_size]+IMAGE_SIZE)
      return {image:batch_image,label_:batch_label}
    else:
      for i in range(batch_size):
        temp=[]
        randnum=random.randint(0,self.Test_Size-1)
        temp.append(self.Test['image'][randnum])
        batch_image.append(temp)
        temp=[0,0,0,0,0,0,0,0,0,0]
        temp[int(self.Test['label'][randnum])]=1
        batch_label.append(temp)
      batch_image=np.reshape(batch_image,[batch_size]+IMAGE_SIZE)
      return {image:batch_image,label_:batch_label}
  def __init__(self,ROOT,batch_index=[1,2,3,4,5],load_test=1):
    """ load all of cifar """
    xs = [] # list
    ys = []
    # 训练集batch 1～5
    for b in batch_index:
      f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
      X, Y = self.__load_CIFAR_batch(f)
      xs.append(X) # 在list尾部添加对象X, x = [..., [X]]
      ys.append(Y)    
      self.Train_Size+=10000
    self.Train['image'] = np.concatenate(xs) # [ndarray, ndarray] 合并为一个ndarray
    self.Train['label'] = np.concatenate(ys)
    del X, Y
    # 测试集
    if load_test==1:
      self.Test['image'] , self.Test['label'] = self.__load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
      self.Test_Size=10000

#卷积神经网络
def CNN_interface(image):
    with tf.variable_scope("Layer{Layer}_{Type}".format(Layer=1,Type="conv")):
        layer1_filter=tf.get_variable("layer1_filter",shape=LAYER1_FILTER_SIZE,initializer=tf.random_normal_initializer(stddev=0.1))
        layer1_biase=tf.get_variable("Layer1_biase",shape=LAYER1_BIASE_SIZE,initializer=tf.constant_initializer(0.0))
        
        layer1=tf.nn.conv2d(image,layer1_filter,strides=LAYER1_STRIDES,padding='SAME')
        biase1=tf.nn.bias_add(layer1,layer1_biase)
        relu1=tf.nn.relu(biase1)
    
    with tf.variable_scope("Layer{Layer}_{Type}".format(Layer=2,Type="pool")):
        layer2=tf.nn.max_pool(relu1,ksize=LAYER2_FILTER_SIZE,strides=LAYER2_STRIDES,padding='SAME')
        relu2=tf.nn.relu(layer2)

    with tf.variable_scope("Layer{Layer}_{Type}".format(Layer=3,Type="conv")):
        layer3_filter=tf.get_variable("layer3_filter",shape=LAYER3_FILTER_SIZE,initializer=tf.random_normal_initializer())
        layer3_biase=tf.get_variable("Layer3_biase",shape=LAYER3_BIASE_SIZE,initializer=tf.constant_initializer(0.0))
        
        layer3=tf.nn.conv2d(relu2,layer3_filter,strides=LAYER3_STRIDES,padding='SAME')
        biase3=tf.nn.bias_add(layer3,layer3_biase)
        relu3=tf.nn.relu(biase3)

    with tf.variable_scope("Layer{Layer}_{Type}".format(Layer=4,Type="pool")):
        layer4=tf.nn.max_pool(relu3,ksize=LAYER4_FILTER_SIZE,strides=LAYER4_STRIDES,padding='SAME')
        relu4=tf.nn.relu(layer4)

    pool_shape=relu4.get_shape().as_list()
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped=tf.reshape(relu4,[-1,nodes],name='reshaped')
    print(pool_shape[0],nodes)
    return reshaped

#全连接神经网络
def Forward_network_interface(data,Layer):
    #建立权重
    with tf.variable_scope("Layer{Layer}_{Type}".format(Layer=5,Type="fulllink")):
        weight=[]
        for i in range(len(Layer)-1):
            weight.append(tf.Variable(tf.random_normal(shape=[Layer[i],Layer[i+1]],stddev=0.01,dtype=tf.float32)))
            tf.add_to_collection("fc_weight",weight[i])
    #计算
    depth=len(Layer)
    for i in range(depth-1):
        print(input,weight[i])
        if i==0:
            result=tf.nn.relu(tf.matmul(data,weight[i]))
        else:
            result=tf.nn.relu(tf.matmul(result,weight[i]))
    return result


image=tf.placeholder(tf.float32,shape=[None]+IMAGE_SIZE,name='image')
label_=tf.placeholder(tf.float32,shape=[None,Layer[-1]],name='label_')

line_shape_result=CNN_interface(image)
label=Forward_network_interface(line_shape_result,Layer)
label_to_num=tf.argmax(label,1)[0]

#学习率指数衰减
learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_DECAY_STEP,LEARNING_RATE_DECAY_RATE)
#交叉熵
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_,logits=label),name='loss')
#正则化
regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
regularization=0.0
weight=tf.get_collection("fc_weight")
for i in weight:
    regularization+=regularizer(i)
#损失
loss=regularization+cross_entropy
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

#正确率
crrect_prediction=tf.equal(tf.argmax(label,1),tf.argmax(label_,1))
accuracy=tf.reduce_mean(tf.cast(crrect_prediction,tf.float64))

saver=tf.train.Saver()

with tf.Session() as sess:
    print("==================SET TIMER==================")
    reply=input('Set time limit(s, unlimited: -1): ')
    TIMER=int(reply)
    print("=====================LOAD====================")
    #初始化
    reply=input('Load model?(y/n): ')
    if reply=='y':
        saver.restore(sess,'./Tensorflow_model/model.ckpt')
    else:
        sess.run(tf.global_variables_initializer())
    print("=================Chose Mode==================")
    dataset=CIFAR10(DATASET_FOLDER)
    reply=input('Start test?(y:test): ')
    if reply=='y':
        for i in range(len(dataset.Test['image'])):
            output=sess.run(label_to_num,feed_dict={image:np.reshape(dataset.Test['image'][i],[1]+[IMAGE_SIZE])})
            print(i+1,',',output,sep='')
    print("===================RESULT====================")
    accuracy_test_count=0

    time_start=time.time()
    for i in range(TRAINING_STEPS):
        #计时器
        time_check=time.time()
        if (time_check-time_start)>TIMER | TIMER!=-1:
            break
        #训练
        feed_dict_train=dataset.nextbatch(BATCH_SIZE)
        sess.run(train_step,feed_dict=feed_dict_train)
        #测试
        if i%CHECK_FREQUNCY==0:
            if accuracy_test_count==0:
                accuracy_test_dict=dataset.nextbatch(ACCURACY_TEST_BATCHSIZE,type='Test')
                accuracy_test_count=ACCURACY_TEST_DATA_UPDATE_FREQUENCY
            accuracy_test_count-=1
            #print("Training step:",i,'Accuracy:',sess.run(accuracy,feed_dict=feed),sess.run(loss,feed_dict=feed),sess.run(learning_rate),answer(sess.run(y,feed_dict=feed)[0]),answer(sess.run(y_,feed_dict=feed)[0]))
            #print("Steps:",i,'Accuracy:',"%.1f" % sess.run(accuracy,feed_dict=accuracy_test_dict),'Learning rate:',sess.run(learning_rate),sess.run(regularization,feed_dict=accuracy_test_dict),sess.run(cross_entropy,feed_dict=accuracy_test_dict))
            print(">> Steps:{step: <5} Accuracy:{acc:.2%}\n   Learning rate:{lr:.5f} Cross entropy:{ce:.5f} Regularization:{re:.5f}".format(step=i,acc=sess.run(accuracy,feed_dict=accuracy_test_dict),lr=sess.run(learning_rate),ce=sess.run(cross_entropy,feed_dict=accuracy_test_dict),re=sess.run(regularization,feed_dict=accuracy_test_dict)))
    writer=tf.summary.FileWriter("./log",tf.get_default_graph())
    writer.close()
    print("===================================")
    reply=input('Save?(y/n): ')
    if reply=='y':
        saver.save(sess,'./Tensorflow_model/model.ckpt')
    print("===================================")

pass