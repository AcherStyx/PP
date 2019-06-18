import matplotlib.pyplot as plt 
import numpy as np 
import mapcreater as mapper
import cv2.cv2 as cv


def pickthemin(map,a):
    if a==[]:
        print("Enpty")
        return False
    
    minone=a[0]
    current=a[0]
    F=map.attach[current[0]][current[1]].H+map.attach[current[0]][current[1]].G

    for current in a:
        if map.attach[current[0]][current[1]].H+map.attach[current[0]][current[1]].G < F:
            minone=current
            F=map.attach[current[0]][current[1]].H+map.attach[current[0]][current[1]]
    
    return minone

def A_star(map):
    openlist=[]
    openlist.append([0,0])
    while True:
        current=pickthemin(map,openlist)
        for i in range(3):
            for ii in range(3):
                if not(i==1 and ii==1):
                    i-=1
                    ii-=1
                    if map.attach[current+i][current+ii]:
                        pass
        
    
    
if __name__ == "__main__":
    a=mapper.map()
    while True:
        a.rand(0.2)
        
    