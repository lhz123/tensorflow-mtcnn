import tensorflow as tf
slim = tf.contrib.slim
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d,global_avg_pool,global_max_pool
def entry_flow(input_data):
    net = slim.conv2d(input_data,32,[3,3],2)
    net1 = slim.conv2d(net,64,[3,3])
    net1_1 = slim.conv2d(net1,128,[1,1],2)
    net = slim.separable_conv2d(net1,128,[3,3],depth_multiplier=1)
    net = slim.separable_conv2d(net, 128, [3, 3],depth_multiplier=1)
    net = slim.max_pool2d(net,[2,2],2)
    net2 = tf.add(net,net1_1)#对应元素相加
    net2_2 = slim.conv2d(net2,256,[2,2],2)
    net = slim.separable_conv2d(net2, 256, [3, 3],depth_multiplier=1)
    net = slim.separable_conv2d(net, 256, [3, 3],depth_multiplier=1)
    net = slim.conv2d(net,256,[2,2],2)
    net3 = tf.add(net,net2_2)
    net3_3 = slim.conv2d(net3,256,[1,1],2)
    net = slim.separable_conv2d(net3, 256, [3, 3],depth_multiplier=1)
    net = slim.separable_conv2d(net, 256, [3, 3],depth_multiplier=1)
    net = slim.conv2d(net,256,[1,1],2)
    net3 = tf.add(net, net3_3)
    return net3

def middle_flow(input_data):
    net = slim.separable_conv2d(input_data, 728, [3, 3],depth_multiplier=1)
    net = slim.separable_conv2d(net, 728, [3, 3],depth_multiplier=1)
    net = slim.separable_conv2d(net, 256, [3, 3],depth_multiplier=1)
    net = tf.add(net, input_data)
    return net

def exit_flow(input_data,num_classes):
    net = slim.separable_conv2d(input_data, 728, [3, 3],depth_multiplier=1)
    net = slim.separable_conv2d(net, 1024, [3, 3],depth_multiplier=1)

    net = slim.conv2d(net,1024,[2,2],2)

    net1 = slim.conv2d(input_data,1024,[1,1],2)
    net = tf.add(net,net1)
    net = slim.separable_conv2d(net, 1536, [3, 3],depth_multiplier=1)
    net = slim.separable_conv2d(net, 2048, [3, 3],depth_multiplier=1)
    net = global_avg_pool(net)
    net = slim.fully_connected(net,num_classes)

    net = slim.softmax(net)
    return net

def inference(input_data,num_class=128):
    net = entry_flow(input_data)
    for i in range(6):
        net = middle_flow(net)

    net = exit_flow(net,num_class)

    return net












