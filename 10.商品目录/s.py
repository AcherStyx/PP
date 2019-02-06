#人生苦短，请使用Python

database=[]

def showmenue():
    print('**********************************')
    print('0.打印全部')
    print('1.添加项目')
    print('2.删除项目')
    print('3.清空全部')
    print('4.价格排序')
    print('5.查找项目')
    print('6.修改项目')
    print('9.退出')
    print('**********************************')

def read_all(a):
    '''
    将输入储存到a中
    '''
    a['name']=input('输入商品名称: ')
    a['id']=input('输入商品id: ')
    a['price']=input('输入商品价格: ')

def print_all(database):
    for temp in database:
        print('**********************************************************')
        print('name: {name}     id: {id}     price: {price}'.format_map(temp))
        print('**********************************************************')

def add(database):
    '''
    添加一个新项目
    '''
    temp={}.fromkeys(['name','id','price'])
    read_all(temp)
    database.append(temp)

def delete(database):
    chose=input('要删除的商品id: ')
    for i in database:
        if chose==i['id']:
            database.remove(i)
            print('成功删除')
            return
    print('未找到该项目')

def clear(database):
    for i in database[:]:
        database.remove(i)

def gen_key(temp):
    return int(temp['price'])

def search(database):
    se=input('输入要查找的商品id: ')
    for i in database:
        if se==i['id']:
            print('**********************************************************')
            print('name: {name}     id: {id}     price: {price}'.format_map(i))
            print('**********************************************************')
            return
    print('未找到该项目')

def modify(database):
    id=input('输入要修改的商品id: ')
    for i in database:
        if id==i['id']:
            database.remove(i)
            temp={}.fromkeys(['name','id','price'])
            read_all(temp)
            database.append(temp)


while 1:
    showmenue()
    chose=int(input('输入选择: '))
    if chose==0:
        print_all(database)
    elif chose==1:
        add(database)
    elif chose==2:
        delete(database)
    elif chose==3:
        clear(database)
    elif chose==4:
        database.sort(key=gen_key)
    elif chose==5:
        search(database)
    elif chose==6:
        modify(database)
    else:
        break