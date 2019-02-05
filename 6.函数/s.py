#懒惰是一种美德

#简单定义一个求和函数
def isum(listin):
    temp=0
    for x in listin:
        temp+=x
    return temp
testdata=[1,2,3,4,5,6,7,8,9,10]
print(isum(testdata))

#尝试修改1
testdata1=123
testdata2=321
def change1(a,b):
    c=a
    a=b
    b=c
change1(testdata1,testdata2)
print(testdata1,testdata2)

#尝试修改2
testdata1=[123,]
testdata2=[321,]
change1(testdata1,testdata2)
print(testdata1,testdata2)

#尝试修改3
def change3(a,b):
    a[0]+=1
    b[0]+=1
change3(testdata1,testdata2)
print(testdata1,testdata2)

#尝试修改4
change3(testdata1[:],testdata2[:])
print(testdata1,testdata2)

'''
总结：
对于普通的变量，函数不能修改其值
对于序列，传给函数的不是一个副本，而是一个链接，和指针类似
能修改原来的数据，这时候的修改会对函数外部的变量造成影响
'''