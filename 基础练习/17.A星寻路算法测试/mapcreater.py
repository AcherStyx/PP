import numpy as np 
import matplotlib.pyplot as plt
import random as rd
import sys 
sys.setrecursionlimit(1000000000)
from time import sleep
import cv2.cv2 as cv

class node:
    '''
    0 未使用过
    1 打开
    2 关闭
    '''
    H=float("inf")
    G=float("inf")
    F=H+G
    states=0
    father=None


class map:
    '''
    states:
    0 空节点
    1 障碍
    5 起点、终点
    '''
    map=[]
    attach=[]
    father=[]
    start=None
    end=None
    height=None
    width=None
    way=[]
    firstsleep=0
    def __init__(self,size=[50,50],start=[0,0],end=[49,49]):
        self.height=size[0]
        self.width=size[1]

        self.map=[[0 for i in range(self.width)] for i in range(self.height)]
        self.father=[[None for i in range(self.width)] for i in range(self.height)]
        self.set_start_end(start,end)
        self.attach=[[node() for i in range(self.width)] for ii in range(self.height)]
    def set_start_end(self,start,end,fill_only=False):
        if fill_only==True:
            self.map[self.start[0]][self.start[1]]=5
            self.map[self.end[0]][self.end[1]]=5
            return

        self.start=start
        self.end=end
        self.map[self.start[0]][self.start[1]]=5
        self.map[self.end[0]][self.end[1]]=5
#        self.print()
    def rand(self,block_percentage=0.5,clr=True):
        if clr==True:
            self.clear()
        for i in range(self.height):
            for ii in range(self.width):
                chance=rd.randint(0,100)+100*block_percentage
                if chance>100:
                    self.map[i][ii]=1
                else:
                    self.map[i][ii]=0
        self.set_start_end(None,None,fill_only=True)
    def set_father(self,current,father):
        self.father[current[0]][current[1]]=father
    def clear(self):
        #self.map=[[0 for i in range(self.width)] for i in range(self.height)]
        self.father=[[None for i in range(self.width)] for i in range(self.height)]
        self.way=[]
    btree_size=0
    def has_way(self,current=None):
        if current==None:
            self.btree_size=0
#            plt.imshow(self.map)
#            plt.pause(0.001)
            current=self.start
        if current==self.end:
#            plt.imshow(self.map)
#            plt.pause(1)
            self.way=self.map
            return True
        if self.map[current[0]][current[1]]!=0 and self.map[current[0]][current[1]]!=5:
            return False

        if self.btree_size>=5000:
            return False

        self.map[current[0]][current[1]]=4
#        print("==")
#        self.print()
        self.firstsleep+=1
    #    if self.firstsleep%10==0:
        #plt.figure(1)
        #plt.imshow(self.map) 
        #plt.ion()
        #plt.axis('off') # 不显示坐标轴
        #plt.show()
        #plt.pause(0.0000001)
        #plt.close()

        #plt.close()
        image=np.array([self.map]*3,dtype=np.uint8)
        image=np.transpose(image,[1,2,0])
        for i in range(self.height):
            for ii in range(self.width):
                if image[i,ii,0]==0:
                    image[i,ii,:]=[255,255,255]
                elif image[i,ii,0]==1:
                    image[i,ii,:]=[0,0,0]
                elif image[i,ii,0]==4:
                    image[i,ii,:]=[0,0,255]
                else:
                    image[i,ii,:]=[100,100,100]
                

        image=np.array(image,dtype=np.uint8)
       # print(image)
        cv.namedWindow('image', cv.WINDOW_GUI_EXPANDED | cv.WINDOW_KEEPRATIO)
        cv.imshow("image",image)
        cv.waitKey(1)


        try:
            if self.has_way([current[0],current[1]+1]):
                return True
        except IndexError:
            pass
        try:
            if self.has_way([current[0]+1,current[1]]):
                return True
        except IndexError:
            pass
        
        if current[0]-1>0 and self.has_way([current[0]-1,current[1]]):
            return True
        if current[1]-1>0 and self.has_way([current[0],current[1]-1]):
            return True
        
        self.btree_size+=4
        
        self.map[current[0]][current[1]]=2
        return False
    def print(self):
        for row in self.map:
            print(row)

if __name__ == "__main__":
    a=map()
    a.rand(0.4)
    #for row in a.map:
    #    print(row)
    #plt.imshow(a.map) #空格为紫色，障碍为黄色
    #plt.pause(3600)
    while True:
        #print("==============")
        a.rand(0.4)
        #plt.imshow(a.map)
        #plt.pause(0.01)
        if a.has_way()==True:
            #plt.imshow(a.way)
            #plt.pause(0.5)
            #plt.clf()
            
            pass
    cv.waitKey()
    print("exit?")
        #a.print()
    