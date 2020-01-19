import numpy as np
import math
import matplotlib.pyplot as plt

N=200

# 内部函数
def f(x):
    return 30*np.sin(x)+20*np.cos(10*x)+10*np.sin(20*x)+np.cos(30*x)

# 傅里叶变化公式
def F(mu,x,N):
    F_mu = [] 
    for mu_n in mu:
        sum = 0
        for x_n in x:
            sum+=f(x_n)*np.math.exp(-2*math.pi*x_n*mu_n*complex(0,1)/N)
        F_mu.append(sum)

    return np.array(np.abs(F_mu))

# 离散的x取值
x = np.linspace(0,math.pi,N)
# 离散的\mu取值
mu = np.linspace(0,8,200)

result_f = f(x)

result_F = F(mu,x,N)

# 子图 1 
ax1 = plt.subplot(1,2,1)
ax1.set_title("f(x_n)")
ax1.plot(x,result_f,"g",label='f(x_n)')
# 子图 2
ax2 = plt.subplot(1,2,2)
ax2.set_title("|F(\mu_n)|")
ax2.plot(x,result_F,"r",label='F(\mu_n)')

plt.show()
