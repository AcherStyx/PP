import tensorflow as tf

def CNN_Interface(data,Layers,lineshape=True,bias=True,order=0,stddev=0.1,const_init=0.0):
    '''
    传入初始的数据，以及按格式创建的各个层的参数，就可以完成卷积层和池化层的创建
    Layers示例:
    CNN_LAYERS=[
        ["conv",[3,3,3,32],[1,1,1,1],True,"SAME"],
        ["pool",[1,8,8,1],[1,8,8,1]],
        ]
    卷积层：卷积层标识|过滤器大小|过滤器移动步长|是否使用偏置项|边界处理方式|
    池化层：池化层标识|池化层大小|池化层移动步长|
    其他参数：
    order 当前是第几次条用本函数，多以调用需要更改order以防止变量名重复
    stddev 卷积层初始化参数
    const_init 偏置项初始化值
    '''
    hidden_layer=data
    with tf.variable_scope("Convolutional_Neural_Networks_{order}".format(order=order)):
        for layer_order,layer in enumerate(Layers):
            if layer[0]=="conv":
                #过滤器
                cnn_filter=tf.get_variable("conv{layer_order}_filter".format(layer_order=layer_order),shape=layer[1],initializer=tf.random_normal_initializer(stddev=stddev))
                tf.summary.histogram("conv{layer_order}_filter".format(layer_order=layer_order), cnn_filter)
                temp_layer=tf.nn.conv2d(hidden_layer,cnn_filter,layer[2],padding="SAME")
                if bias==True:
                    #偏置项
                    cnn_bias=tf.get_variable("conv{layer_order}_biase".format(layer_order=layer_order),shape=[layer[1][3]],initializer=tf.constant_initializer(const_init))
                    tf.summary.histogram("conv{layer_order}_filter".format(layer_order=layer_order), cnn_bias)
                    temp_bias=tf.nn.bias_add(temp_layer,cnn_bias)
                    hidden_layer=tf.nn.relu(temp_bias)
                else:
                    hidden_layer=tf.nn.relu(temp_layer)
            else:
                temp_layer=tf.nn.max_pool(hidden_layer,ksize=layer[1],strides=layer[2],padding="SAME")
                hidden_layer=tf.nn.relu(temp_layer)
    if lineshape==True:
        pool_shape=hidden_layer.get_shape().as_list()
        nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
        reshaped=tf.reshape(hidden_layer,[-1,nodes],name='reshaped')
        return reshaped,nodes
    else:
        return hidden_layer

def FC_Interface(data,Layer,order=0,keep_prob_layer=1,keep_prob_image=1,stddev=0.1,const_init=0.0):
    #建立权重
    with tf.variable_scope("Fully_Lincked_Networks_{order}".format(order=order)):
        weight=[]
        bias=[]
        for i in range(len(Layer)-1):
            weight.append(tf.get_variable("fc{layer_order}_weight".format(layer_order=i),shape=[Layer[i],Layer[i+1]],initializer=tf.random_normal_initializer(stddev=stddev)))
            bias.append(tf.get_variable("fc{layer_order}_bias".format(layer_order=i),shape=[Layer[i+1]],initializer=tf.constant_initializer(const_init)))
            #加入到集合以供计算正则化
            tf.add_to_collection("fc_weight",weight[i])
            #添加到Tensorboard
            tf.summary.histogram("fc{layer_order}_filter".format(layer_order=i), weight[i])
            #tf.summary.histogram("fc{layer_order}_bias".format(layer_order=i), bias[i])
    #计算
    data=tf.nn.dropout(data,keep_prob=keep_prob_image)
    for i in range(len(Layer)-1):
        if i==0:
            result=tf.nn.relu(tf.matmul(data,weight[i]))
        else:
            result_droped=tf.nn.dropout(result,keep_prob=keep_prob_layer)
            result=tf.nn.relu(tf.matmul(result_droped,weight[i]))
    return result


if __name__ == "__main__":
    print("=========================")
    import random
    import numpy as np
    TEST_LAYERS=[
        ["conv",[5,5,3,8],[1,1,1,1],True,"SAME"],
        ["pool",[1,2,2,1],[1,2,2,1]],
        ["conv",[5,5,8,32],[1,1,1,1],True,"SAME"],
        ["pool",[1,2,2,1],[1,2,2,1]],
    ]
    
    image=tf.Variable(tf.random_normal([5,32,32,3],mean=1.0))
    print(image)
    result,index=CNN_Interface(image,TEST_LAYERS)
    print(result.shape)
    FC_LAYER=[index,100,10]
    fcout=FC_Interface(result,FC_LAYER)
    print(fcout.shape)
    print("=========================")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(result))