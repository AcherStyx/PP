import random
import math

class Digits:
    __label=[]
    __image=[]
    __all_dis=[]
    #读入训练数据
    def __Get_TrainingDigits(self):
        file_name="./trainingDigits/{num}_{index}.txt"
        for num in range(10):
            index=0
            while 1:
                try:
                    current_file=open(file_name.format(num=num,index=index))
                    self.__label.append(num)
                    current_image=current_file.read(32*33-1).split('\n')
                    current_image=[countlen(x) for x in current_image]
                    self.__image.append(current_image)
                    index+=1
                except:
                    break
    #计算距离
    def __Distance(self,image_0):
        self.__all_dis=[]
        for image_1 in self.__image:
            dis=0
            for i,ii in zip(image_0,image_1):
                dis+=(i-ii)**2
            self.__all_dis.append(dis)
    #查找最大的n个数的下标
    def __Find_N_Min(self,n):
        index=[]
        for ii in range(n):
            min=1000000000
            min_index=0
            indexcount=0
            for i in self.__all_dis:
                if indexcount not in index:
                    if i<min:
                        min=i
                        min_index=indexcount
                indexcount+=1
            index.append(min_index)
        return index
    #求出给定一组index对应的出现最多的label
    def __Find_Answer(self,index):
        answer=[0]*10
        for i in index:
            answer[self.__label[i]]+=(1.0/self.__all_dis[i])**2
        return answer.index(max(answer))
    #识别
    def Recognize(self,image):
        #在第一次读入数据集
        if self.__image==[]:
            self.__Get_TrainingDigits()

        self.__Distance(image)
        index=self.__Find_N_Min(5)
        answer=self.__Find_Answer(index)
                        
        return answer



def countlen(line):
    line=[int(i) for i in line]
    return sum(line)
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
def Get_TestDigits_Image():
    image=[]
    file_name="./testDigits/{num}_{index}.txt"
    for num in range(10):
        index=0
        while 1:
            try:
                current_file=open(file_name.format(num=num,index=index))
                current_image=current_file.read(32*33-1)
                image.append(current_image)
                index+=1
            except:
                break
    return image

test_label,test_image=Get_TestDigits()
test_show=Get_TestDigits_Image()
Digits_Recognizer=Digits()

total=0
right=0
for i in range(1000):
    num=random.randint(0,len(test_label))
    #print(test_show[num])
    answer=Digits_Recognizer.Recognize(test_image[num])
    if test_label[num]==answer:
        right+=1
    total+=1
    print("预测: ",answer,"答案: ",test_label[num],"正确率: ",float(right/total*100))
    