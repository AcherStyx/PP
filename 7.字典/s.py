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
#copy
diceg1=data.copy()
diceg2=data
print(diceg1)
print(diceg2)
diceg1['extra']=0
print(data)
diceg2['extra']=0
print(data) #若使用直接复制，则替换时也会改变原件
#clear
diceg1.clear()
print(diceg1)
#fromkeys
eg={}.fromkeys(['name','phone','adress'],'N/A')
print(eg)
eg={}
eg.fromkeys(['name','phone','adress'],'N/A') #fromkeys不就地修改字典
print(eg)
#get
eg={}.fromkeys(['name','phone','adress'],'N/A')
print(eg.get('name'))
print(eg.get('N/A'))
#item
print(eg.items()) #返回一个可迭代的字典视图
eg['name']='Kaede'
print(eg.items()) #会同步修改
#key
print(eg.keys())
#pop
eg.pop('adress')
print(eg)
#popitem
eg.popitem() #随机弹出一个
print(eg)
#setdefault
eg.setdefault('phone','N/A')
print(eg)
eg.setdefault('name','N/A') #键已经存在时，不会更改
print(eg)
#update
change={'name':'qb','adress':'hok'}
eg.update(change)
print(eg)
#value
print(eg.values())
print(list(eg.values()))