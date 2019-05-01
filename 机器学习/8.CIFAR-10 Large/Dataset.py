import numpy as np 
from six.moves import cPickle as pickle
import os
import platform
import random

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
            batch_image=np.reshape(batch_image,[batch_size]+[32,32,3])
            return batch_image,batch_label
        else:
            for i in range(batch_size):
                temp=[]
                randnum=random.randint(0,self.Test_Size-1)
                temp.append(self.Test['image'][randnum])
                batch_image.append(temp)
                temp=[0,0,0,0,0,0,0,0,0,0]
                temp[int(self.Test['label'][randnum])]=1
                batch_label.append(temp)
            batch_image=np.reshape(batch_image,[batch_size]+[32,32,3])
            return batch_image,batch_label
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
        
if __name__ == "__main__":
    data=CIFAR10('./dataset/')

    pass