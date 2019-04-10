import _thread as thread
from time import sleep,ctime
import requests

dataset=["http://github.com/",
"http://www.google.com",
"http://www.baidu.com/",
"http://www.bilibili.com/",
"http://dasd.donit.com/"
"http://www.baidu.com/fdsfsdfsdfsdfas/"
]

def connect(urllink):
    try:
        r=requests.get(urllink)
        print('[',r.status_code,']',urllink)
    except Exception as e:
        print('[',e,']',urllink)

def pause(data,times,lock):
    print("No. {N} start at {TIME}".format(N=times,TIME=ctime()))
    connect(data)
    print("No. {N} end at {TIME}".format(N=times,TIME=ctime()))
    if times==0:
        pass
    else:
        while lock[times-1].locked:
            pass
    lock[times].release()

def main():
    locks=[]
    for i in dataset:
        lock=thread.allocate_lock()
        lock.acquire()
        locks.append(lock)
    
    print("Start at {TIME}".format(TIME=ctime()))

    for index , data in enumerate(dataset):
        thread.start_new_thread(pause,(data,index,locks))

    for lock in locks:
        while(lock.locked()):
            pass

    print("End at {TIME}".format(TIME=ctime()))

if __name__=='__main__':
    main()
