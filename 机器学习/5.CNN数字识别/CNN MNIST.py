import tensorflow as tf
import csv
import random
import time
import numpy

BATCH_SIZE=50
TRAINING_STEPS=50000
#频率
CHECK_FREQUNCY=100
ACCURACY_TEST_DATA_UPDATE_FREQUENCY=5
#学习率设置
LEARNING_RATE_BASE=0.01
LEARNING_RATE_DECAY_RATE=0.99
LEARNING_RATE_DECAY_STEP=1000
#正则化比率
REGULARIZATION_RATE=0.005


#层1 卷积层
LAYER1_FILTER_SIZE=[5,5,1,4]
LAYER1_BIASE_SIZE=[LAYER1_FILTER_SIZE[3]]
LAYER1_STRIDES=[1,1,1,1]
#层2 池化层
LAYER2_FILTER_SIZE=[1,2,2,1]
LAYER2_STRIDES=[1,2,2,1]
#层3 卷积层
LAYER3_FILTER_SIZE=[5,5,4,16]
LAYER3_BIASE_SIZE=[LAYER3_FILTER_SIZE[3]]
LAYER3_STRIDES=[1,1,1,1]
#层4 池化层
LAYER4_FILTER_SIZE=[1,2,2,1]
LAYER4_STRIDES=[1,2,2,1]
#层1-? 全连接层
Layer=(784,200,100,10) 

#总计训练次数
global_step=tf.Variable(0,trainable=False)

#数据集封装
class Kaggle_Train_Digits:
    label=[]
    image=[]
    def read_data(self,datatype='Train'):
        if datatype=='Train':
            self.label=[]
            self.image=[]
            with open('./train.csv') as datafile:
                train = [[float (x) for x in row] for row in list(csv.reader(datafile,delimiter=','))[1:]]
            for image in train:
                self.label.append(image[0])
                self.image.append(image[1:])
            pass
        else:
            self.label=[]
            self.image=[]
            with open('./test.csv') as datafile:
                test = [[float (x) for x in row] for row in list(csv.reader(datafile,delimiter=','))[1:]]
            for image in test:
                self.image.append(image[:])
        return
    def show_image(self,index):
        print(self.label[index])
        for i in range(28):
            for ii in range(28):
                if 0!=self.image[index][i*28+ii]:
                    print(1,end='')
                else:
                    print(0,end='')
            print(" ")
    def nextbatch(self,BATCH_SIZE=None):
        batch_image=[]
        batch_label=[]
        if len(self.label)==len(self.image):
            for i in range(BATCH_SIZE):
                temp=[]
                randnum=random.randint(0,len(self.label)-1)
                temp.append(self.image[randnum])
                batch_image.append(temp)
                temp=[0,0,0,0,0,0,0,0,0,0]
                temp[int(self.label[randnum])]=1
                batch_label.append(temp)
            batch_image=numpy.reshape(batch_image,[BATCH_SIZE,28,28,1])
            return {image:batch_image,label_:batch_label}
        else:
            batch_image=numpy.reshape(image,[len(image),28,28,1])
            return {image:batch_image}

#卷积神经网络 前向传播
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


image=tf.placeholder(tf.float32,shape=[None,28,28,1],name='image')
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
    dataset=Kaggle_Train_Digits()
    reply=input('Start train or test?(0:test): ')
    if reply=='0':
        dataset.read_data('Test')
        for i in range(len(dataset.image)):
            output=sess.run(label_to_num,feed_dict={image:numpy.reshape(dataset.image[i],[1,28,28,1])})
            print(i+1,',',output,sep='')
    else:
        #读入训练数据
        dataset.read_data('Train')
        print("===================SAMPLE====================")
        dataset.show_image(random.randint(0,len(dataset.image)))
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
                accuracy_test_dict=dataset.nextbatch(1000)
                accuracy_test_count=ACCURACY_TEST_DATA_UPDATE_FREQUENCY
            accuracy_test_count-=1
            #print("Training step:",i,'Accuracy:',sess.run(accuracy,feed_dict=feed),sess.run(loss,feed_dict=feed),sess.run(learning_rate),answer(sess.run(y,feed_dict=feed)[0]),answer(sess.run(y_,feed_dict=feed)[0]))
            #print("Steps:",i,'Accuracy:',"%.1f" % sess.run(accuracy,feed_dict=accuracy_test_dict),'Learning rate:',sess.run(learning_rate),sess.run(regularization,feed_dict=accuracy_test_dict),sess.run(cross_entropy,feed_dict=accuracy_test_dict))
            print("Steps:{step: <5} Accuracy:{acc:.1%}\nLearning rate:{lr:.5f} Cross entropy:{ce:.5f} Regularization:{re:.5f}".format(step=i,acc=sess.run(accuracy,feed_dict=accuracy_test_dict),lr=sess.run(learning_rate),ce=sess.run(cross_entropy,feed_dict=accuracy_test_dict),re=sess.run(regularization,feed_dict=accuracy_test_dict)))
    writer=tf.summary.FileWriter("./log",tf.get_default_graph())
    writer.close()
    print("===================================")
    reply=input('Save?(y/n): ')
    if reply=='y':
        saver.save(sess,'./Tensorflow_model/model.ckpt')
    print("===================================")