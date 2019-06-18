import random as rd 

for i in range(1000):
    if(i % 1000==0):
        print("%5d" % rd.randint(999,1000),sep=' ',end=' ')
    print("%5d" % rd.randint(0,1000),sep=' ',end=' ')
#    print("%5d" % rd.randint(0,10),sep=' ',end=' ')
#    print("%5d" % rd.randint(0,10),sep=' ',end=' ')
#    print("%05d" % rd.randint(0,10000),sep=' ',end=' ')