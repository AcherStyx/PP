from six.moves import cPickle as pickle
import numpy as np 
import os
import platform

import tensorflow as tf
import random
import time

import NeuralNetwork as net

BATCH_SIZE=5
TRAINING_STEPS=20000
DATASET_FOLDER='./dataset/'
#检测频率
CHECK_FREQUNCY=1000
ACCURACY_TEST_DATA_UPDATE_FREQUENCY=5
ACCURACY_TEST_BATCHSIZE=400
#学习率设置
LEARNING_RATE_BASE=1e-5
LEARNING_RATE_DECAY_RATE=0.98
LEARNING_RATE_DECAY_STEP=100
#正则化比率
#REGULARIZATION_RATE=0.001
REGULARIZATION_RATE=0.00000000001
#REGULARIZATION_RATE=0.000005

DROPOUT_FORWARD_NETWORK=1

IMAGE_SIZE=[32,32,3]

#层1-? 全连接层
Layer=[128,512,128,64,10]
#总计训练次数
global_step=tf.Variable(0,trainable=False)

CNN_LAYERS=[
    ["conv",[3,3,3,32],[1,1,1,1],True,"SAME"],
    ["conv",[2,2,32,48],[1,1,1,1],True,"SAME"],
    ["conv",[2,2,48,48],[1,1,1,1],True,"SAME"],
    ["pool",[1,2,2,1],[1,2,2,1]],
    ["conv",[3,3,48,80],[1,1,1,1],True,"SAME"],
    ["pool",[1,2,2,1],[1,2,2,1]],
    ["conv",[3,3,80,128],[1,1,1,1],True,"SAME"],
    ["pool",[1,8,8,1],[1,8,8,1]],
]

#输出格式串
OUTPUT_FORMAT_STRING=">> Steps:{step: <5} Learning rate:{lr:.10f} Cross entropy:{ce:.5f} Regularization:{re:.5f}\n   Accuracy:{acc:.2%} \n   Accuracy in Train data:{accit:.2%}\n{srlt}"

#数据集读取
class CIFAR10:
    '''摘自：https://www.cnblogs.com/jimobuwu/p/9161531.html
    自己稍作封装
    官网的示例代码在encodeing那里有问题
    '''
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
            return {image:batch_image,label_:batch_label,keep_prob_holder:DROPOUT_FORWARD_NETWORK}
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
            return {image:batch_image,label_:batch_label,keep_prob_holder:1}
    def testdict(self):
        batch_label=[]
        for i in range(self.Test_Size):
            temp=[0,0,0,0,0,0,0,0,0,0]
            temp[int(self.Test['label'][i])]=1
            batch_label.append(temp)
        return {image:self.Test['image'],label_:batch_label}
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

keep_prob_holder=tf.placeholder(tf.float32)
image=tf.placeholder(tf.float32,shape=[None]+IMAGE_SIZE,name='image')
label_=tf.placeholder(tf.float32,shape=[None,Layer[-1]],name='label_')

line_shape_result,len_shape_length=net.CNN_Interface(image,CNN_LAYERS,bias=False)
Layer[0]=len_shape_length
label=net.FC_Interface(line_shape_result,Layer,keep_prob_layer=keep_prob_holder)

label_to_num=tf.argmax(label,1)[0]
label__to_num=tf.argmax(label_,1)[0]
label_sampel=tf.nn.softmax(label[0])

#学习率指数衰减
learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_DECAY_STEP,LEARNING_RATE_DECAY_RATE)
#交叉熵
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_,logits=label),name='loss')
#正则化
regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
regularization_i=0.0
weight=tf.get_collection("fc_weight")
for i in weight:
    regularization_i+=regularizer(i)
#损失
regularization=regularization_i
#loss=regularization+cross_entropy
loss=cross_entropy
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

#正确率
crrect_prediction=tf.equal(tf.argmax(label,1),tf.argmax(label_,1))
accuracy=tf.reduce_mean(tf.cast(crrect_prediction,tf.float64))

saver=tf.train.Saver()


with tf.Session() as sess:
    summaries = tf.summary.merge_all()
    writer=tf.summary.FileWriter("./log",tf.get_default_graph())
    print("=====================INFO====================")
    print(\
    "Learning rate base:{lrb}\n"\
    "Learning rate decay rate:{lrdr}\n"\
    "Learning rate decay step:{lrds}\n"
    "Regularization rate:{regu}\n"\
    "Keep prob:{drop}"\
    .format(lrb=LEARNING_RATE_BASE,regu=REGULARIZATION_RATE,drop=DROPOUT_FORWARD_NETWORK,lrdr=LEARNING_RATE_DECAY_RATE,lrds=LEARNING_RATE_DECAY_STEP))
    print("==================SET TIMER==================")
    reply=input('Set time limit(s, unlimited: -1): ')
    try:
        TIMER=int(reply)
    except ValueError:
        TIMER=-1
    print("====================LOAD=====================")
    #初始化
    reply=input('Load model?(y/n): ')
    if reply=='y':
        saver.restore(sess,'./Tensorflow_model/model.ckpt')
    else:
        sess.run(tf.global_variables_initializer())
    print("=================INPUT DATA==================")
    dataset=CIFAR10(DATASET_FOLDER)
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
        del feed_dict_train
        #测试
        if i%CHECK_FREQUNCY==0:
            if accuracy_test_count==0:
                accuracy_test_train_dict=dataset.nextbatch(ACCURACY_TEST_BATCHSIZE)
                accuracy_test_dict=dataset.nextbatch(ACCURACY_TEST_BATCHSIZE,type='Test')
                accuracy_test_count=ACCURACY_TEST_DATA_UPDATE_FREQUENCY
            accuracy_test_count-=1
            #print("Training step:",i,'Accuracy:',sess.run(accuracy,feed_dict=feed),sess.run(loss,feed_dict=feed),sess.run(learning_rate),answer(sess.run(y,feed_dict=feed)[0]),answer(sess.run(y_,feed_dict=feed)[0]))
            #print("Steps:",i,'Accuracy:',"%.1f" % sess.run(accuracy,feed_dict=accuracy_test_dict),'Learning rate:',sess.run(learning_rate),sess.run(regularization,feed_dict=accuracy_test_dict),sess.run(cross_entropy,feed_dict=accuracy_test_dict))
            print(OUTPUT_FORMAT_STRING.format(\
                step=i,\
                acc=sess.run(accuracy,feed_dict=accuracy_test_dict),\
                lr=sess.run(learning_rate),\
                ce=sess.run(cross_entropy,feed_dict=accuracy_test_dict),\
                re=sess.run(regularization,feed_dict=accuracy_test_dict),\
                accit=sess.run(accuracy,feed_dict=accuracy_test_train_dict),\
                srlt=sess.run(label_sampel,feed_dict=accuracy_test_dict)),\
                sess.run(label_to_num,feed_dict=accuracy_test_dict),\
                sess.run(label__to_num,feed_dict=accuracy_test_dict))
            #写Tensorboard信息
            summ = sess.run(summaries, feed_dict=accuracy_test_dict)
            writer.add_summary(summ, global_step=i)
            net.Matedata_Writer(writer,accuracy_test_train_dict,train_step,i)
    writer.close()
    print("===================================")
    reply=input('Save?(y/n): ')
    if reply=='y':
        saver.save(sess,'./Tensorflow_model/model.ckpt')
    print("===================================")