import random
import math

class Digits():
    label=None
    image=None
        
#计算距离
def distance(image_0,image_1):
    dis=0
    for i,ii in zip(image_0,image_1):
        dis+=(i-ii)**2
    #return math.sqrt(dis)
    return dis
#查找最大的n个数的下标
def find_n_min(list,n):
    index=[]
    for ii in range(n):
        min=1000000000
        min_index=0
        indexcount=0
        for i in list:
            if indexcount not in index:
                if i<min:
                    min=i
                    min_index=indexcount
            indexcount+=1
        index.append(min_index)
    return index
def find_answer(index,label,dis):
    answer=[0]*10
    for i in index:
        answer[label[i]]+=(1.0/dis[i])**2
    return answer.index(max(answer))
#计算每行1的数量
def countlen(line):
    line=[int(i) for i in line]
    return sum(line)
#读入训练数据
def Get_TrainingDigits():
    label=[]
    image=[]
    file_name="./trainingDigits/{num}_{index}.txt"
    for num in range(10):
        index=0
        while 1:
            try:
                current_file=open(file_name.format(num=num,index=index))
                label.append(num)
                current_image=current_file.read(32*33-1).split('\n')
                current_image=[countlen(x) for x in current_image]
                image.append(current_image)
                index+=1
            except:
                break
    return label,image
#读入测试数据
def Get_TestDigits():
    label=[]
    image=[]
    file_name="./testDigits/{num}_{index}.txt"
    for num in range(10):
        index=0
        while 1:
            try:
                current_file=open(file_name.format(num=num,index=index))
                label.append(num)
                current_image=current_file.read(32*33-1).split('\n')
                current_image=[countlen(x) for x in current_image]
                image.append(current_image)
                index+=1
            except:
                break
    return label,image

#读入
Train=Digits()
Test=Digits()
Train.label,Train.image=Get_TrainingDigits()
Test.label,Test.image=Get_TestDigits()

total=0
right=0
index=0
for right_answer,image_0 in zip(Test.label,Test.image):
    find_distance=[]
    for image_1 in Train.image:
        find_distance.append(distance(image_0,image_1))
    index=find_n_min(find_distance,10)
    my_answer=find_answer(index,Train.label,find_distance)
    if right_answer==my_answer:
        right+=1
    total+=1
    if total%100==0:
        print(float(right)/total)

pass