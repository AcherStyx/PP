import struct
import numpy as np
import random
filename='./train-images.idx3-ubyte'

TRAIN_STEP=5



class ReadFromMNIST:
    image=[]
    image2=[]
    numImages=numRows=numColumns=0
    def __init__(self,path):
        binfile=open(filename,'rb')
        buf=binfile.read()
        index=0
        magic,self.numImages,self.numRows,self.numColumns=struct.unpack_from('>IIII',buf,index)
        index+=struct.calcsize('>IIII')
        for im in range(0,self.numImages):
            im=struct.unpack_from('>784B',buf,index)
            index+=struct.calcsize('>784B')
            im=np.array(im,dtype='uint8')
            self.image2.append(im)
            im=im.reshape(28,28)
            im=[sum(x) for x in im]
            self.image.append(im)
    def nextbatch(self,size):
        batch=[]
        index=[]
        for i in range(size):
            index.append(random.randint(0,self.numImages-1))
            batch.append(self.image[index[i]])
        return batch,index
    def showimage(self,index):
        im=self.image2[index]
        for row in range(self.numRows):
            for col in range(self.numColumns):
                print(int(bool(im[row*self.numColumns+col])),end='')
            print('')

def Pick_Random_Cs(num_of_c):
    c=[]
    image=[]
    for i in range(num_of_c):
        c.append(np.random.randint(0,254,size=[784]))
    return c

def Cal_Dist(Cs,batch):
    nearest_index=[]
    for image in batch:
        dist=[]
        for c in Cs:
            dist.append(np.dot(image,c)/(np.linalg.norm(image)*np.linalg.norm(c)))
        nearest_index.append(dist)
    return nearest_index

def Update_Cs(batch,dist):
    Cs=[]
    nearest=np.argmin(dist,axis=0)
    for item in nearest:
        Cs.append(batch[item])
    return Cs

dataset=ReadFromMNIST(filename)
#0 质心初始化
Cs=dataset.image[0:20]
for i in range(TRAIN_STEP):
    #1 计算点到质心距离。分类
    batch,index=dataset.nextbatch(2000)
    dist=Cal_Dist(Cs,batch)
    #2 更新质心
    Cs=Update_Cs(batch,dist)

q='1'
while q!='n':
    batch,index=dataset.nextbatch(100)
    dist=Cal_Dist(Cs,batch)
    classified=[np.argmin(x) for x in dist]
    for i in range(20):
        print('c',i)
        for ii,c in enumerate(classified):
            if c == i:
                dataset.showimage(index[ii])
    q=input("Continue?")

pass