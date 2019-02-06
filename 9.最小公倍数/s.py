temp_a=a=int(input('num 1: '))
temp_b=b=int(input('num 2: '))


while temp_a != temp_b:
    if temp_a>temp_b:
        temp_b+=b
    elif temp_a<temp_b:
        temp_a+=a

print(temp_a)


