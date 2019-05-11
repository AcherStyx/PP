import csv
import random
import os
import matplotlib.pyplot as plt
import struct
import numpy as np 

# 训练集文件
train_images_idx3_ubyte_file = 'train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'train-labels.idx1-ubyte'
# 测试集文件
test_images_idx3_ubyte_file = 't10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 't10k-labels.idx1-ubyte'

class MNIST_Dataset():
    train={}
    test={}
    train_batch_index=0
    test_batch_index=0
    def __read_image(self,path_to_file):
        bin_data = open(path_to_file, 'rb').read()
            # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
        offset=0
        fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
        magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
        #print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

        # 解析数据集
        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
        #print(offset)
        fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
        #print(fmt_image,offset,struct.calcsize(fmt_image))
        images = np.empty((num_images, num_rows, num_cols))
        #plt.figure()
        for i in range(num_images):
            #if (i + 1) % 10000 == 0:
                #print('已解析 %d' % (i + 1) + '张')
                #print(offset)
            images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
            #print(images[i])
            offset += struct.calcsize(fmt_image)
            #plt.imshow(images[i],'gray')
            #plt.pause(0.00001)
            #plt.show()
        #plt.show()
        return images
    def __read_label(self,path_to_file):
        # 读取二进制数据
        bin_data = open(path_to_file, 'rb').read()

        # 解析文件头信息，依次为魔数和标签数
        offset = 0
        fmt_header = '>ii'
        magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
        #print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

        # 解析数据集
        offset += struct.calcsize(fmt_header)
        fmt_image = '>B'
        labels = np.empty(num_images)
        for i in range(num_images):
            #if (i + 1) % 10000 == 0:
                #print ('已解析 %d' % (i + 1) + '张')
            labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
            offset += struct.calcsize(fmt_image)
        return labels
    def __init__(self,root_path):
        #train image
        file_path=os.path.join(root_path,train_images_idx3_ubyte_file)
        self.train["image"]=self.__read_image(file_path)
        file_path=os.path.join(root_path,train_labels_idx1_ubyte_file)
        self.train["label"]=self.__read_label(file_path)
        file_path=os.path.join(root_path,test_images_idx3_ubyte_file)
        self.test["image"]=self.__read_image(file_path)
        file_path=os.path.join(root_path,test_labels_idx1_ubyte_file)
        self.test["label"]=self.__read_label(file_path)
    def showsample(self,dataset="train",amount=5,index=None,pause=0.5):
        if index!=None:
            if dataset=="train":
                plt.figure()
                plt.imshow(self.train["image"][index],"gray")
                plt.text(1, -0.6, "Number: {num}  Label: {label}".format(num=int(self.train["label"][index]),label=index), fontsize=15)
                plt.show()
                return
            elif dataset=="test":
                plt.figure()
                plt.imshow(self.test["image"][index],"gray")
                plt.text(1, -0.6, "Number: {num}  Label: {label}".format(num=int(self.test["label"][index]),label=index), fontsize=15)
                plt.show()
                return
            else:
                raise(ValueError)
        if dataset=="train":
            for i in range(amount):
                index=random.randint(0,60000)
                plt.ion()
                plt.figure(i)
                plt.imshow(self.train["image"][index],"gray")
                plt.text(1, -0.6, "Number: {num}  Label: {label}".format(num=int(self.train["label"][index]),label=index), fontsize=15)
                plt.show()
                plt.pause(0.1)
            plt.pause(pause)
            for i in range(amount):
                plt.close(i)
        elif dataset=="test":
            for i in range(amount):
                index=random.randint(0,10000)
                plt.ion()
                plt.figure(i)
                plt.imshow(self.test["image"][index],"gray")
                plt.text(1, -0.6, "Number: {num}  Label: {label}".format(num=int(self.test["label"][index]),label=index), fontsize=15)
                plt.show()
                plt.pause(0.07)
            plt.pause(0.5)
            for i in range(amount):
                plt.close(i)
        else:
            raise(ValueError)
    def shuffle(self):
        seed=random.randint(0,10000)
        np.random.seed(seed)
        np.random.shuffle(self.train["image"])
        np.random.seed(seed)
        np.random.shuffle(self.train["label"])
        np.random.seed(seed)
        np.random.shuffle(self.test["image"])
        np.random.seed(seed)
        np.random.shuffle(self.test["label"])
    def nextbatch(self,batch_size,set="train"):
        if set=="train":
            start=self.train_batch_index%60000
            end=min(start+batch_size,60000)
            image=self.train["image"][start:end]
            label=self.train["label"][start:end]
            return image,label
        elif set=="test":
            start=self.test_batch_index%10000
            end=min(start+batch_size,10000)
            image=self.test["image"][start:end]
            label=self.test["label"][start:end]
            return image,label
        else:
            raise(ValueError)

if __name__ == "__main__":
    print("++++++++++++++++++++")
    data=MNIST_Dataset("./MNIST")
    print("Load data successfully")
    print("++++++++++++++++++++")
    print("Sample: before shuffle")
    #data.showsample(index=1000,dataset="test",amount=20)
    print("Shuffle")
    data.shuffle()
    print("Sample: after shuffle")
    #data.showsample(index=1000,dataset="test",amount=20)
    print("++++++++++++++++++++")
    print("shape: ",data.train["image"][0].shape)
    print("++++++++++++++++++++")
    data.train_batch_index=59997
    print(data.nextbatch(10)[0].shape)
    data.test_batch_index=9997
    print(data.nextbatch(10,set="test")[0].shape)
    print("++++++++++++++++++++")
    pass
