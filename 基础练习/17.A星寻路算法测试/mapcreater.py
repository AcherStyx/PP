import numpy as np 
import matplotlib.pyplot as plt
import random as rd

class map:
    map=[]
    father=[]
    start=None
    end=None
    height=None
    width=None
    way=[]
    def __init__(self,size=[100,100],start=[0,0],end=[99,99]):
        self.height=size[0]
        self.width=size[1]

        self.map=[[0 for i in range(self.width)] for i in range(self.height)]
        self.father=[[None for i in range(self.width)] for i in range(self.height)]
        self.set_start_end(start,end)
    def set_start_end(self,start,end,fill_only=False):
        if fill_only==True:
            self.map[self.start[0]][self.start[1]]=3
            self.map[self.end[0]][self.end[1]]=3
            return

        self.start=start
        self.end=end
        self.map[self.start[0]][self.start[1]]=3
        self.map[self.end[0]][self.end[1]]=3
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
        if self.map[current[0]][current[1]]!=0 and self.map[current[0]][current[1]]!=3:
            return False

        if self.btree_size>=1000:
            return False

        self.map[current[0]][current[1]]=2
#        print("==")
#        self.print()
#        plt.imshow(self.map)
#        plt.pause(0.000001)
        
        try:
            if self.has_way([current[0]+1,current[1]]):
                return True
        except IndexError:
            pass
        try:
            if self.has_way([current[0],current[1]+1]):
                return True
        except IndexError:
            pass
        
        if current[0]-1>0 and self.has_way([current[0]-1,current[1]]):
            return True
        if current[1]-1>0 and self.has_way([current[0],current[1]-1]):
            return True
        
        self.btree_size+=4
        
        self.map[current[0]][current[1]]=0
        return False
    def print(self):
        for row in self.map:
            print(row)

if __name__ == "__main__":
    a=map()
    a.rand(0.4)
    for row in a.map:
        print(row)
    #plt.imshow(a.map) #空格为紫色，障碍为黄色
    #plt.pause(3600)
    while(1):
        #print("==============")
        a.rand(0.3)
        plt.imshow(a.map)
        plt.pause(0.01)
        if a.has_way()==True:
            plt.imshow(a.way)
            plt.pause(0.5)
        #a.print()