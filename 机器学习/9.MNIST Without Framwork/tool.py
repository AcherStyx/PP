import numpy as np 


def active(x):
    y=[]
    for i in x:
        if i>0:
            y.append(i)
        else:
            y.append(0)
    return np.array(y)

def softmax(y):
    y=y-np.array([max(y)]*len(y))
    return np.exp(y)/np.sum(np.exp(y))

def cross_ecropy(y,y_):
    if sum(y)>=1.00001:
        raise(ValueError)
    return -sum(y_*np.log(y))

def loss(y,y_):
    softmaxed=softmax(y)
    return cross_ecropy(softmaxed,y_)

def loss_mintwo(y,y_):
    return 0.5*np.sqrt((y-y_)*(y-y_))

def create_weight(LAYER):
    w=[]
    for i in range(len(LAYER)-1):
        w.append(np.random.randn(LAYER[i],LAYER[i+1])/2)
    return w

def forward(weight,x):
    for w in weight:
        x=x.dot(w)
        x=active(x)
    return x

#grade

def grade_active(z):
    l=[]
    for row in z:
        ll=[]
        for col in row:
            if col>0:
                ll.append(col)
            else:
                ll.append(0.0)
        l.append(ll)
    return l

def grade_l(weight,grade_l1,grade_active):
    wT=np.transpose(weight)
    t=wT.cdot(grade_l1)
    return t*grade_active

def grade_output():
    pass

if __name__ == "__main__":
    print("++++++++++++++++++++\n++++++++++++++++++++")
    x=np.array([2.0,5.,3.,5.,6.])
    y=[10.0,10.0,5.0,5.0]
    y_=[1.0,0,0,0]
    y_softmaxed=softmax(y)
    print(y_softmaxed)
    print(sum(y_softmaxed))

    cross_ecropy=loss(y_softmaxed,y_)
    print(cross_ecropy)

    #Tensorflow
    if False:
        import tensorflow as tf
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        print("++++++++++++++++++++\n++++++++++++++++++++")
        y_tfsoftmaxed=tf.nn.softmax(y)
        ce_tf=tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
        sess=tf.InteractiveSession()
        print("++++++++++++++++++++\n++++++++++++++++++++")
        print(sess.run(y_tfsoftmaxed))
        print(sess.run(ce_tf))
    
    print("++++++++++++++++++++\n++++++++++++++++++++")
    LAYER=(5,2000,1000,500,100,10) 
    w=create_weight(LAYER)
    for i in range(len(LAYER)-1):
        print(w[i].shape)
    print("++++++++++++++++++++\n++++++++++++++++++++")
    y=output=forward(w,x)
    print(y)
        