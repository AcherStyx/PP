import random as rd 

file = open("data.txt","w")

for i in range(100000):
    if(i % 10==0):
        file.write("1 ")
        file.write("%5d " % rd.randint(1,12))
    file.write("%5d " % rd.randint(0,1000))
    file.write("%5d " % rd.randint(0,1000))
    file.write("%5d " % rd.randint(0,1000))
    file.write("%5d " % rd.randint(0,1000))
    file.write("%5d " % rd.randint(0,1000))
    file.write("%5d " % rd.randint(0,1000))
    file.write("%5d \n" % rd.randint(0,1000))

file.close()
    