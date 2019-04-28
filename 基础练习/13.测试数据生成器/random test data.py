import random as rd 

for i in range(10000):
    if(i % 10==0):
        print("%5d" % rd.randint(0,12),sep=' ',end=' ')
    print("%5d" % rd.randint(0,10000),sep=' ',end=' ')
    print("%5d" % rd.randint(0,10000),sep=' ',end=' ')
    print("%5d" % rd.randint(0,10000),sep=' ',end=' ')
#    print("%05d" % rd.randint(0,10000),sep=' ',end=' ')