from six.moves import cPickle as pickle
import numpy as np 
import os
import platform
import matplotlib.pyplot as plt 

import tensorflow as tf
import random
import time

import NeuralNetwork as net

BATCH_SIZE=2
TURN=0
TRAINING_STEPS=200000
DATASET_FOLDER='./dataset/'
#检测频率
CHECK_FREQUNCY=500
ACCURACY_TEST_DATA_UPDATE_FREQUENCY=1
ACCURACY_TEST_BATCHSIZE=400
#学习率设置
LEARNING_RATE_BASE=1.02839185e-05
LEARNING_RATE_DECAY_RATE=0.99
LEARNING_RATE_DECAY_STEP=5000
#正则化比率
#REGULARIZATION_RATE=0.001
REGULARIZATION_RATE=0.0001
#REGULARIZATION_RATE=0.000005

DROPOUT_CNN_NETWORK=0.9

IMAGE_SIZE=[32,32,3]
#层1-? 全连接层
Layer=[0,512,128,10]

#输出格式串
OUTPUT_FORMAT_STRING=">> Steps:{step: <5} Learning rate:{lr:.10f}\n   Cross entropy:{ce:.25f} Regularization:{re:.5f}\n   Accuracy:{acc:.2%} \n   Accuracy in Train data:{accit:.2%}\n{srlt}"

#数据集读取
class CIFAR10:
    '''摘自：https://www.cnblogs.com/jimobuwu/p/9161531.html
    稍作封装
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
            datadict = self.__load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
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
            return {image:batch_image,label_:batch_label,keep_prob_holder:DROPOUT_CNN_NETWORK}
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

with tf.variable_scope("Input_Data"):
    #输入数据
    keep_prob_holder=tf.placeholder(tf.float32,name='keep_prob_cnn')
    image=tf.placeholder(tf.float32,shape=[None]+IMAGE_SIZE,name='image')
    label_=tf.placeholder(tf.float32,shape=[None,Layer[-1]],name='label_')

with tf.variable_scope("Neural_Network"):
    #卷积层
    CNN_LAYERS=[
    ["conv",[3,3,3,32],[1,1,1,1],0.1,"SAME",False],
    ["conv",[3,3,32,32],[1,1,1,1],0.1,"SAME",False],
    ["conv",[3,3,32,32],[1,1,1,1],0.1,"SAME",False],
    ["conv",[3,3,32,48],[1,1,1,1],0.1,"SAME",False],
    ["conv",[3,3,48,48],[1,1,1,1],0.1,"SAME",False],
    ["maxpool",[1,4,4,1],[1,2,2,1]],
    ["dropout",keep_prob_holder],
    ["conv",[3,3,48,80],[1,1,1,1],0.1,"SAME",False],
    ["conv",[3,3,80,80],[1,1,1,1],0.1,"SAME",False],
    ["conv",[3,3,80,80],[1,1,1,1],0.1,"SAME",False],
    ["conv",[3,3,80,80],[1,1,1,1],0.1,"SAME",False],
    ["conv",[3,3,80,80],[1,1,1,1],0.1,"SAME",False],
    ["avgpool",[1,4,4,1],[1,2,2,1]],
    ["dropout",keep_prob_holder],
    ["conv",[3,3,80,128],[1,1,1,1],0.1,"SAME",False],
    ["conv",[3,3,128,128],[1,1,1,1],0.1,"SAME",False],
    ["conv",[3,3,128,128],[1,1,1,1],0.1,"SAME",False],
    ["conv",[3,3,128,128],[1,1,1,1],0.1,"SAME",False],
    ["conv",[3,3,128,128],[1,1,1,1],0.1,"SAME",False],
    ["avgpool",[1,8,8,1],[1,4,4,1]],
    #["dropout",keep_prob_holder]
    ]
    line_shape_result,len_shape_length=net.CNN_Interface(image,CNN_LAYERS,bias=False,active_function=tf.nn.relu,output_each_layer=True)
    Layer[0]=len_shape_length
    label=net.FC_Interface(line_shape_result,Layer,keep_prob_layer=keep_prob_holder,stddev=0.001,active_function=tf.nn.relu)

with tf.variable_scope("Sample_Output"):
    #测试输出
    label_to_num=tf.argmax(label,1)[0]
    label__to_num=tf.argmax(label_,1)[0]
    label_sampel=tf.nn.softmax(label[0])

with tf.variable_scope("Train_Model"):
    #总计训练次数
    global_step=tf.Variable(0,trainable=False)
    #学习率
    LEARNING_RATE_BASE=tf.Variable(LEARNING_RATE_BASE)
    #学习率指数衰减
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_DECAY_STEP,LEARNING_RATE_DECAY_RATE)
    #损失
    loss,cross_entropy,regularization=net.Softmax_Cross_Encropy_With_Regularization(label_,label,REGULARIZATION_RATE)
    #训练
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

with tf.variable_scope("Analyze_And_Save"):
    #正确率
    crrect_prediction=tf.equal(tf.argmax(label,1),tf.argmax(label_,1))
    accuracy_testdict=tf.reduce_mean(tf.cast(crrect_prediction,tf.float64),name="Accuracy_In_Test")
    accuracy_traindict=tf.reduce_mean(tf.cast(crrect_prediction,tf.float64),name="Accuracy_In_Train")
    tf.summary.scalar("Accuracy_In_Test",accuracy_testdict)
    #储存
    saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #写文件
    summaries = tf.summary.merge_all()
    writer=tf.summary.FileWriter("./log",tf.get_default_graph())
    print("=====================INFO====================")
    net.Info_Printer(learning_rate_base=sess.run(LEARNING_RATE_BASE),
    learning_rate_decay_rate=LEARNING_RATE_DECAY_RATE,
    learning_rate_decay_step=LEARNING_RATE_DECAY_STEP,
    regularization_rate=REGULARIZATION_RATE,
    keep_prob_layer=DROPOUT_CNN_NETWORK
    )
    print("CNN output",len_shape_length)
    print("==================SET TIMER==================")
    timer=net.TrainTimer()
    timer.getinput()
    print("====================LOAD=====================")
    #初始化
    reply_load=input('Load model?(y/n): ')
    if reply_load=='y':
        saver.restore(sess,'./Tensorflow_model/model.ckpt')
    print("==============TRY LEARNING RATE==============")
    reply=input("Try Learning rate?(y/n): ")
    print("=================INPUT DATA==================")
    dataset=CIFAR10(DATASET_FOLDER)
    if reply=="y":
        if reply_load=='y':
            net.Learning_Rate_Search(LEARNING_RATE_BASE,train_step,accuracy_testdict,dataset,1e-6,lr_upper_bond=1e-3,lr_raise_rate=1.5,train_step=4000,restore='./Tensorflow_model/model.ckpt',print_data=True)
        else:
            net.Learning_Rate_Search(LEARNING_RATE_BASE,train_step,accuracy_testdict,dataset,1e-6,lr_upper_bond=1e-3,lr_raise_rate=1.5,train_step=4000,print_data=True)
        reply=input("Input a new learning rate: ")
        sess.run(tf.global_variables_initializer())
        new_lr=tf.assign(LEARNING_RATE_BASE,float(reply))
        sess.run(new_lr)
    
    net.bar("Result")
    accuracy_test_count=0
    time_start=time.time()
    #lr_manual=tf.assign(LEARNING_RATE_BASE,1e-05)
    #sess.run(lr_manual)
    for i in range(TRAINING_STEPS): 
        try:
            i=i+TURN*TRAINING_STEPS
            #计时器
            if timer.check==1:
                break
            #训练a
            feed_dict_train=dataset.nextbatch(BATCH_SIZE)
            sess.run(train_step,feed_dict=feed_dict_train)
            #测试
            if i%CHECK_FREQUNCY==0:
                if accuracy_test_count==0:
                    accuracy_test_train_dict=dataset.nextbatch(ACCURACY_TEST_BATCHSIZE)
                    accuracy_test_dict=dataset.nextbatch(ACCURACY_TEST_BATCHSIZE,type='Test')
                    accuracy_test_count=ACCURACY_TEST_DATA_UPDATE_FREQUENCY
                accuracy_test_count-=1
                #print("Training step:",i,'Accuracy:',sess.run(accuracy_testdict,feed_dict=feed),sess.run(loss,feed_dict=feed),sess.run(learning_rate),answer(sess.run(y,feed_dict=feed)[0]),answer(sess.run(y_,feed_dict=feed)[0]))
                #print("Steps:",i,'Accuracy:',"%.1f" % sess.run(accuracy_testdict,feed_dict=accuracy_test_dict),'Learning rate:',sess.run(learning_rate),sess.run(regularization,feed_dict=accuracy_test_dict),sess.run(cross_entropy,feed_dict=accuracy_test_dict))
                print(OUTPUT_FORMAT_STRING.format(\
                    step=i,\
                    acc=sess.run(accuracy_testdict,feed_dict=accuracy_test_dict),\
                    lr=sess.run(learning_rate),\
                    ce=sess.run(cross_entropy,feed_dict=accuracy_test_dict),\
                    re=sess.run(regularization,feed_dict=accuracy_test_dict),\
                    accit=sess.run(accuracy_traindict,feed_dict=accuracy_test_train_dict),\
                    srlt=sess.run(label_sampel,feed_dict=accuracy_test_dict)),\
                    sess.run(label_to_num,feed_dict=accuracy_test_dict),\
                    sess.run(label__to_num,feed_dict=accuracy_test_dict))
                #写Tensorboard信息
                summ = sess.run(summaries, feed_dict=accuracy_test_dict)
                writer.add_summary(summ, global_step=i)
                #net.Matedata_Writer(writer,accuracy_test_dict,train_step,i)
        except KeyboardInterrupt:
            net.bar("Pause")
            new_lr=input("New learning rate base(-1 to quit): ")
            try:
                if float(new_lr)>0:
                    try:
                        set_init=input("Refresh Model?(y/n): ")
                        if set_init=="y":
                            sess.run(tf.global_variables_initializer())
                        change_lr=tf.assign(LEARNING_RATE_BASE,float(new_lr))
                        sess.run(change_lr)
                    except Exception as e:
                        print("[!]Failed to change learning rate!")
                        print(e)
                    net.bar("Continue")
                else:
                    break
            except:
                pass
    writer.close()
    print("===================================")
    reply=input('Save?(y/n): ')
    if reply=='y':
        saver.save(sess,'./Tensorflow_model/model.ckpt')
    print("===================================")