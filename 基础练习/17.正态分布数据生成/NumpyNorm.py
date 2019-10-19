import numpy as np
import matplotlib.pyplot as plt
import CSVToFile as csv

x = np.linspace(-10,10,1000)

y = (1/(np.sqrt(2*np.pi)))*np.exp(-x**2/2)

plt.plot(x,y)

plt.xlim(-5,5)
plt.ylim(-0.1,1)
plt.grid(True)

plt.legend()
plt.show()


x = x.reshape([-1,1])
y = y.reshape([-1,1])

point = np.concatenate([x,y],axis=1)

csv.CSVToFile(point.tolist(),"data.csv",1)