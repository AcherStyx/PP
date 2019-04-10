import _thread as thread
from time import sleep,ctime

procedures=[10,5,6,4,8,7,5,2,1,4,6,5,7,4,9]

def pause(sleeptime,times,lock):
    print("No. {N} start at {TIME}".format(N=times,TIME=ctime()))
    sleep(sleeptime)
    print("No. {N} end at {TIME}".format(N=times,TIME=ctime()))
    lock.release()

def main():
    locks=[]
    for i in procedures:
        lock=thread.allocate_lock()
        lock.acquire()
        locks.append(lock)
    
    print("Start at {TIME}".format(TIME=ctime()))

    for index , sleeptime in enumerate(procedures):
        thread.start_new_thread(pause,(sleeptime,index,locks[index]))

    for lock in locks:
        while(lock.locked()):
            pass

    print("End at {TIME}".format(TIME=ctime()))

if __name__=='__main__':
    main()