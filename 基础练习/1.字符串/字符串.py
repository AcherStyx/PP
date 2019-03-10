print("Hello World")
print("Hello" " World") #字符串串联
print("Hello"+" World")
print("Hello"+" World"*2) #适合乘法运算
print('Hello World') #单引号
print(2*2)
print(2.0*2)
print(2*2.0) #和C中的类型转换不太一样
x=2
y=2.0
print(int(x)*float(y))


#格式化字符串
print("==========")
file_name="{num}_{index}.txt"
print(file_name.format(num=1,index=10))
