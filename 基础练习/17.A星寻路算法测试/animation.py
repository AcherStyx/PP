
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()   #生成子图，相当于fig = plt.figure(),ax = fig.add_subplot(),其中ax的函数参数表示把当前画布进行分割，例：fig.add_subplot(2,2,2).表示将画布分割为两行两列.ax在第2个子图中绘制，其中行优先，
xdata, ydata = [], []      #初始化两个数组
ln, = ax.plot([], [], 'r-', animated=False)  #第三个参数表示画曲线的颜色和线型，具体参见：https://blog.csdn.net/tengqingyong/article/details/78829596

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,               #返回曲线

def update(n):
    xdata.append(n)         #将每次传过来的n追加到xdata中
    ydata.append(np.sin(n))
    ln.set_data(xdata, ydata)    #重新设置曲线的值
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 1000),     #这里的frames在调用update函数是会将frames作为实参传递给“n”
                    init_func=init, blit=True,interval=1)
plt.show()