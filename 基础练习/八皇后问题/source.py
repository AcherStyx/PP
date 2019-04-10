size=8
map=[[0 for i in range(size)] for i in range(size)]

def abletoplace(place,map):
    for i in range(8):
        if (i<place[0] and map[i][place[1]]) or (place[0]-i>=0 and  ((place[1]-i>=0 and map[place[0]-i][place[1]-i]) or (place[1]+i<8 and map[place[0]-i][place[1]+i]))):
            return 0
    return 1

def solve(map,size=8,line=0):
    count=0
    for i in range(size):
        if abletoplace([line,i],map):
            map[line][i]=1
            if line==size-1:
                for ii in range(size):
                    print(map[ii])
                print("=====")
                map[line][i]=0
                return 1
            count+=solve(map,size=size,line=line+1)
            map[line][i]=0
    return count

print("Output:",solve(map))