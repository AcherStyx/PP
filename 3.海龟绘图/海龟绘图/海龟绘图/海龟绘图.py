from turtle import *
x=0
while x< 0:
    forward(100)
    left(95)
    forward(17.43115)
    left(95)
    forward(100)
    right(180)
    x+=1


#forward(120)

y=0
x=0
delta=0.05
while x<10:
   y=x*(x-5)*(x-1)*(x-5)
   left(90)
   forward(y*3)
   pendown()
   left(180)
   forward(y*3)
   left(90)
   forward(delta*10)
   x+=delta
