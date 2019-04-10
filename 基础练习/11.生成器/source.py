def genera(start,end):
    i=start
    while True:
        yield i
        i+=2
        if i>=end:
            return 

diedaiqi=genera(10,100)
print(list(diedaiqi))

for num in genera(10,200):
    print(num,end=' ')
print('')

#recursion

def slipt(input):
    try:
        input+''
    except TypeError:
        pass
    else:
        raise(TypeError)
    try:
        for elem in input:
            for data in slipt(elem):
                yield data
    except TypeError:
        yield input        


for num in slipt([12,[544,45,7,45,[4],22],45,[12,12]]):
    print(num,end=' ')  
print('')

try:
    for char in slipt("dasdasdasda"):
        print(char,end=' ')
except:
    print("TypeError in function slipt")