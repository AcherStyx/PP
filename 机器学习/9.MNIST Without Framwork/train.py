import csv
import random
import numpy as np 
import mnist_dataset as mnist 
import tool

LAYER=(784,2000,1000,500,100,10) 

#数据输入
#dataset=mnist.MNIST_Dataset("./MNIST/")
#batch_image,batch_label=dataset.nextbatch(1)

x=np.random.rand(784)
y_=np.array([1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
w=tool.create_weight(LAYER)
y=tool.forward(w,x)
print(y)
loss=tool.loss(y,y_)
print("=====")
print(loss)
print("=====")

pass