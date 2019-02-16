import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("/home/ubuntu-user/MNIST/",one_hot=True)

#交叉熵
def cross_entropy(y_,y):
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)

def true_rate(y_,y):
    pass

def answer(arr):
    arr=list(arr)
    index=0
    max=arr[0]
    for i in range(10):
        if arr[i] > max:
            index=i
            max=arr[i]
    return index


lr=tf.Variable(0.0000001)
lr=lr*0.999
x=tf.placeholder(tf.float64,shape=(None,784),name='x')
y_=tf.placeholder(tf.float64,shape=(None,10),name='y_')
w1=tf.Variable(tf.random_normal([784,000],stddev=2,dtype=tf.float64),name='w1')
w2=tf.Variable(tf.random_normal([2000,1000],stddev=2,dtype=tf.float64),name='w2')
w3=tf.Variable(tf.random_normal([1000,500],stddev=2,dtype=tf.float64),name='w3')
w4=tf.Variable(tf.random_normal([500,100],stddev=2,dtype=tf.float64),name='w4')
w5=tf.Variable(tf.random_normal([100,10],stddev=2,dtype=tf.float64),name='w5')

n1=tf.matmul(x,w1)
n1=tf.nn.relu(n1)
n2=tf.matmul(n1,w2)
n2=tf.nn.relu(n2)
n3=tf.matmul(n2,w3)
n3=tf.nn.relu(n3)
n4=tf.matmul(n3,w4)
n4=tf.nn.relu(n4)
y=tf.matmul(n4,w5)
out=tf.nn.softmax(y)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y),name='loss')

train_step=tf.train.GradientDescentOptimizer(lr).minimize(loss)

#常量
batch_size=50
STEP=10000
CHECK=1000
R_CHECK=100
ii=iii=0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(STEP):
        #获得初始化好的字典
        xs,ys=mnist.train.next_batch(batch_size)
        feed={x:xs,y_:ys}
        #训练
        sess.run(train_step,feed_dict=feed)
        #检查训练情况
        if i%(STEP/CHECK)==0:
            xs,ys=mnist.train.next_batch(batch_size)
            feed={x:xs,y_:ys}
            t_y=sess.run(y,feed_dict=feed)[0]
            t_out=ys[0]
            t_y_a=answer(t_y)
            t_out_a=answer(t_out)
            if iii%100==0:
                ii=iii=0
            if t_y_a==t_out_a:
                ii+=1
            iii+=1
            print('E:',i,'正确率',float(ii)/(iii+1)*100,'Loss:',sess.run(loss,feed_dict=feed),t_y_a,t_out_a)
            