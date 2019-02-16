import tensorflow as tf
import csv
import random
import time

#各层节点数
LAYER=(784,2000,1000,500,100,10)   
#单次训练数据量
BATCH_SIZE=200
#频率
CHECK_FREQUNCY=1000
ACCURACY_TEST_DATA_UPDATE_FREQUENCY=10
#学习率设置
LEARNING_RATE_BASE=0.005
LEARNING_RATE_DECAY_RATE=0.99
LEARNING_RATE_DECAY_STEP=1000
#正则化比率
REGULARIZATION_RATE=0.0001
#训练轮数
TRAINING_STEPS=400000
#滑动平均
MOVING_AVERAGE_DECAY=0.99
#总计训练次数
global_step=tf.Variable(0,trainable=False)
#默认退出计时器
TIMER=3600

def get_data_from_Kaggle():
    with open('test.csv') as datafile:
        test = [[float (x) for x in row] for row in list(csv.reader(datafile,delimiter=','))[1:]]
    with open('train.csv') as datafile:
        train = [[float (x) for x in row] for row in list(csv.reader(datafile,delimiter=','))[1:]]
    # 字符串转数字
    return [train,test]

def create_variable(LAYER):
    weight=[]
    for i in range(len(LAYER)-1):
        weight.append(tf.Variable(tf.random_normal(shape=[LAYER[i],LAYER[i+1]],stddev=0.01,dtype=tf.float64)))
    return weight

def forward_network(input,layer,weight,avg_class=None):   #前向传播
    depth=len(layer)
    if avg_class == None:
        for i in range(depth-1):
            if i==0:
                temp=tf.nn.relu(tf.matmul(input,weight[i]))
            else:
                temp=tf.nn.relu(tf.matmul(temp,weight[i]))
    else:
        for i in range(depth-1):
            pass
    return temp

def generate_dict(BATCH_SIZE,data,include_labels=True,labels=None):
    train=[]
    label=[]
    if include_labels==True:
        for i in range(BATCH_SIZE):
            randnum=random.randint(0,len(data)-1)
            train.append(data[randnum][1:])
            temp=[0,0,0,0,0,0,0,0,0,0]
            temp[int(data[randnum][0])]=1
            label.append(temp)
    else:
        if label!=None:
            for i in range(BATCH_SIZE):
                randnum=random.randint(0,len(data)-1)
                train.append(data[randnum][:])
                label.append([labels[randnum],])
    return {x:train,y_:label}


def generate_test_dict(data):
    return {x:data}

def answer(arr):
    arr=list(arr)
    index=0
    max=arr[0]
    for i in range(10):
        if arr[i] > max:
            index=i
            max=arr[i]
    return index

#输入
x=tf.placeholder(tf.float64,[None,LAYER[0]],name='x')
y_=tf.placeholder(tf.float64,[None,LAYER[-1]],name='y_')

#初始化权重
weight=create_variable(LAYER)
#前向传播
y=forward_network(x,LAYER,weight)
out=tf.argmax(y,1)
#学习率指数衰减
learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_DECAY_STEP,LEARNING_RATE_DECAY_RATE)
#交叉熵
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y),name='loss')
#正则化
regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
regularization=0.0
for i in range(len(weight)):
    regularization+=regularizer(weight[i])
#损失函数
loss=cross_entropy+regularization
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
#正确率
crrect_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(crrect_prediction,tf.float64))*100

#持久化
saver=tf.train.Saver()

with tf.Session() as sess:
    print("===============START===============")
    #初始化
    reply=input('Load?(y/n): ')
    if reply=='y':
        saver.restore(sess,'/home/ubuntu-user/MNIST/model.ckpt')
    else:
        sess.run(tf.global_variables_initializer())
    #读入训练数据
    train,test=get_data_from_Kaggle()
    print("=============LOAD TEST=============")
    reply=input('START TEST?(y/n): ')
    if reply=='y':
        for i in range(len(test)):
            output=sess.run(out,feed_dict={x:[test[i],]})
            print(i+1,',',output[0],sep='')
    print("=============SET TIMER=============")
    reply=input('Time limit(s, unlimited: -1): ')
    TIMER=int(reply)
    print("===================================")
    #初始化正确率测试计数器
    accuracy_test_count=0
    time_start=time.time()
    for i in range(TRAINING_STEPS):
        #计时器
        time_check=time.time()
        if (time_check-time_start)>TIMER | TIMER!=-1:
            break
        #创建样本
        feed=generate_dict(BATCH_SIZE,train)
        #训练
        sess.run(train_step,feed_dict=feed)
        #检查
        if i%CHECK_FREQUNCY==0:
            if accuracy_test_count==0:
                accuracy_test_dict=generate_dict(1000,train)
                accuracy_test_count=ACCURACY_TEST_DATA_UPDATE_FREQUENCY
            accuracy_test_count-=1
            #print("Training step:",i,'Accuracy:',sess.run(accuracy,feed_dict=feed),sess.run(loss,feed_dict=feed),sess.run(learning_rate),answer(sess.run(y,feed_dict=feed)[0]),answer(sess.run(y_,feed_dict=feed)[0]))
            print("Training step:",i,'Accuracy:',sess.run(accuracy,feed_dict=accuracy_test_dict),'learning rate:',sess.run(learning_rate))
    print("===================================")
    reply=input('Save?(y/n): ')
    if reply=='y':
        saver.save(sess,'/home/ubuntu-user/MNIST/model.ckpt')
    print("===================================")