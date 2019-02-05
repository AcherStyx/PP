adressbook={'name':'Ini','Phone':123432,'Adress':'Place'}

#通过字典来打印
print('''name: {name} 
Phone: {Phone} 
Adress: {Adress}'''.format_map(adressbook))

#一定要使用三个
#换行直接回车

#添加项
data={}
data['extra']=123
print(data)

#修改项
data['extra']=321
print(data)
data['one']=456
print(data)

#嵌套
data['more']={}
print(data)
data['more']['l2']=123
print(data)

#方法
diceg1=data.copy()
diceg2=data
print(diceg1)
print(diceg2)
diceg1['extra']=0
print(data)
diceg2['extra']=0
print(data) #若使用直接复制，则替换时也会改变原件

