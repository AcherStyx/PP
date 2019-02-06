import time

n=int(input("截止大小: "))

start=time.time()
output=[]
for ii in range(n)[2:]:
    check=1
    for i in range(ii)[2:]:
        if ii%i==0:
            check=0
            break
    if check==1:
        output.append(ii)
end=time.time()

print('耗时: ',(end-start)*1000,'ms')
print(output)