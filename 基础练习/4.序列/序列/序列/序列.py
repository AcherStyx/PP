arr1=["H",1]
arr2=["A",2]
database=[arr1,arr2]
print(arr1[0],arr1[1])
print(database[0][0])
ax=1
print(ax)
print(ax*2)
ax='hghf'#储存数字的变量可以转而储存字符串
print(ax)
print(ax*2)

print("\n切片")
print(database)
numlist=[1,2,3,4,5]
print(numlist)
numlist[2:4]=[7,8]  #切片时，前一个包含在内，后一个不包含在内
print(numlist)
numlist[4:2:-1]=[7,8]   #倒过来时也是前一个包含在内
print(numlist)
print(numlist[4:2:1]);

print(numlist+[1,0,0])
print("8 in numlist:",8 in numlist)
print("9 in numlist:",9 in numlist)

print("\n成员资格检查：")
#成员资格检查
streg="EFGFDSSSSS"
print("E" in streg)
print("GF"in streg)
print("FS" in streg)
streg=["EFGHNN",'FGDD',"DF"]#对于单个字符串和多个字符串构成的序列，成员的意义是不一样的
print('D' in streg)

print('函数:len max min')
print(len(streg))
streg=[23,213,21,21,23]
print(streg)
print("max",max(streg))
print("min",min(streg))

print("基本列表操作")
print(streg)
del streg[1]
print(streg)