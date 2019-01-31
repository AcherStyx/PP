import matplotlib.pyplot as plt
import numpy as np
import math
import csv

#从文件读取数据
with open('data.csv') as datafile:
    data = csv.reader(datafile)
    data=list(data)
#字符串转浮点数
data=[[float (x) for x in row] for row in data]
#拆分到xy变量中
x=[]
y=[]
for row in data:
    x.append(row[0])
    y.append(row[1])
x=np.array(x)
y=np.array(y)
print("数据",'\n',x,'\n',y)

#设置初始值和步长
a=np.array([0.0,0.0])
step=0.00001
#梯度下降
change=1
while change>=0.01:
    change_0=0.0
    change_1=0.0
    for i in range(0,len(x)):
        change_0+=2.0*(a[1]*x[i]+a[0]-y[i])
        change_1+=2.0*x[i]*(a[1]*x[i]+a[0]-y[i])
    a[0]-=change_0*step
    a[1]-=change_1*step
    change=math.sqrt(change_0**2+change_1**2)
print(a[0],a[1])

#求偏导直接计算
n=len(x)
b=np.array([0.0,0.0])
b[1]=(n*sum(x*y)-sum(x)*sum(y))/(n*sum(x*x)-sum(x)*sum(x))
b[0]=sum(y)/n-b[1]*sum(x)/n
print(b[0],b[1])
#二乘法下的误差
current=0
for i in range(0,len(x)):
    current+=(y[i]-a[0]-a[1]*x[i])**2
print("梯度下降:",current)
current=0
for i in range(0,len(x)):
    current+=(y[i]-b[0]-b[1]*x[i])**2
print("直接计算:",current)

#绘图
px=np.linspace(0,max(x),100)
py=px*a[1]+a[0]
_py=px*b[1]+b[0]
plt.plot(x,y,'go',label='data')
plt.plot(px,py,'r',label='predict 1',linewidth=5.0)
plt.plot(px,_py,'b--',label='predict 2')
plt.legend()
plt.show()