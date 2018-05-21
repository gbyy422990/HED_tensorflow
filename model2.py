#coding:utf-8
#Bin GAO

import os
import tensorflow as tf
import numpy as np
import vgg16

learning_rate=1e-5
num_class=2
loss_weight = np.array([1,1])

h = 512   #4032
w = 512   #3024
batch_size = 1
g_mean = [142.53,129.53,120.20]


def unpool2d(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
    with tf.variable_scope(scope):
        input_shape = pool.get_shape().as_list()
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
        return ret


def unet(input,training):
    #归一化[-1,1]
    input = input/127.5 - 1

    pool_parameters = []
    en_parameters = []

    #conv1_1
    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,3,64],dtype=tf.float32,
                                                 stddev=0.1),name='weights')
        conv = tf.nn.conv2d(input,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.1,shape=[64],dtype=tf.float32),
                             trainable=True,name='biases')

        out = tf.nn.bias_add(conv,biases)
        conv1_1 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name=scope)
        en_parameters += [kernel,biases]

    #conv1_2
    with tf.name_scope('conv1_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,64,64],dtype=tf.float32,
                                                 stddev=0.1),name='weights')
        conv = tf.nn.conv2d(conv1_1,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.1,shape=[64],dtype=tf.float32),
                             trainable=True,name='biases')
        out = tf.nn.bias_add(conv,biases)
        conv1_2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name=scope)
        en_parameters += [kernel,biases]

    #branch_e1
    with tf.variable_scope('branch_e1') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 64, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv1_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        branch_e1 = tf.nn.sigmoid(tf.layers.batch_normalization(out, training=training),name='branch_e1')

    #pool1
    pool1,arg1 = tf.nn.max_pool_with_argmax(conv1_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1')
    pool_parameters.append(arg1)

    #conv2_1
    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                 stddev=0.1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')

        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name=scope)
        en_parameters += [kernel, biases]

    #conv2_2
    with tf.name_scope('conv1_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                 stddev=0.1), name='weights')
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name=scope)
        en_parameters += [kernel, biases]

    # branch_e2
    with tf.variable_scope('branch_e2') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 128, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv2_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.image.resize_images(out, size=(h, w), method=tf.image.ResizeMethod.BILINEAR)
        branch_e2 = tf.nn.sigmoid(tf.layers.batch_normalization(out, training=training),name='branch_e2')

    #pool2
    pool2, arg2 = tf.nn.max_pool_with_argmax(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool2')
    pool_parameters.append(arg2)

    # conv3_1
    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name=scope)
        en_parameters += [kernel, biases]

    # conv3_2
    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name=scope)
        en_parameters += [kernel, biases]

    # conv3_3
    with tf.name_scope('conv3_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name=scope)
        en_parameters += [kernel, biases]

    # branch_e3
    with tf.variable_scope('branch_e3') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 256, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.image.resize_images(out,size=(h,w),method=tf.image.ResizeMethod.BILINEAR)
        branch_e3 = tf.nn.sigmoid(tf.layers.batch_normalization(out, training=training),name='branch_e3')

    # pool3
    pool3, arg3 = tf.nn.max_pool_with_argmax(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool3')
    pool_parameters.append(arg3)

    # conv4_1
    with tf.name_scope('conv4_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name=scope)
        en_parameters += [kernel, biases]

    # conv4_2
    with tf.name_scope('conv4_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name=scope)
        en_parameters += [kernel, biases]

    # conv4_3
    with tf.name_scope('conv4_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name=scope)
        en_parameters += [kernel, biases]

    # branch_e4
    with tf.variable_scope('branch_e4') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4_3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.image.resize_images(out, size=(h, w), method=tf.image.ResizeMethod.BILINEAR)
        branch_e4 = tf.nn.sigmoid(tf.layers.batch_normalization(out, training=training),name='branch_e4')

    # pool4
    pool4, arg4 = tf.nn.max_pool_with_argmax(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool4')
    pool_parameters.append(arg4)

    # conv5_1
    with tf.name_scope('conv5_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name=scope)
        en_parameters += [kernel, biases]

    # conv5_2
    with tf.name_scope('conv5_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name=scope)
        en_parameters += [kernel, biases]

    # conv5_3
    with tf.name_scope('conv5_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name=scope)
        en_parameters += [kernel, biases]

    # branch_e5
    with tf.variable_scope('branch_e5') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5_3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.image.resize_images(out, size=(h, w), method=tf.image.ResizeMethod.BILINEAR)
        branch_e5 = tf.nn.sigmoid(tf.layers.batch_normalization(out, training=training),name='branch_e5')

    # pool5
    pool4_1, arg4_1 = tf.nn.max_pool_with_argmax(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool5')
    pool_parameters.append(arg4_1)

    # conv6_1
    with tf.name_scope('conv6_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv6_1 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='conv6_1')
        en_parameters += [kernel, biases]

    # deconv6
    with tf.variable_scope('deconv6') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv6_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        deconv6 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='deconv6')

    # branch_d5
    with tf.variable_scope('branch_d5') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(deconv6, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.image.resize_images(out, size=(h, w), method=tf.image.ResizeMethod.BILINEAR)
        branch_d5 = tf.nn.sigmoid(out,name='branch_d5')

    # deconv5_1/unpooling
    deconv5_1 = unpool2d(deconv6, pool_parameters[-1])

    # deconv5_2
    with tf.variable_scope('deconv5_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(deconv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        deconv5_2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='deconv5_2')

    # branch_d4
    with tf.variable_scope('branch_d4') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(deconv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.image.resize_images(out, size=(h, w), method=tf.image.ResizeMethod.BILINEAR)
        branch_d4 = tf.nn.sigmoid(out,name='branch_d4')

    # deconv4_1/unpooling
    deconv4_1 = unpool2d(deconv5_2, pool_parameters[-2])

    # deconv4_2
    with tf.variable_scope('deconv4_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(deconv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        deconv4_2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='deconv4_2')

    # branch_d3
    with tf.variable_scope('branch_d3') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 256, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(deconv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.image.resize_images(out, size=(h, w), method=tf.image.ResizeMethod.BILINEAR)
        branch_d3 = tf.nn.sigmoid(out,name='branch_d3')

    # deconv3_1/unpooling
    deconv3_1 = unpool2d(deconv4_2, pool_parameters[-3])

    # deconv3_2
    with tf.variable_scope('deconv3_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 256, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(deconv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        deconv3_2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='deconv3_2')

    # branch_d2
    with tf.variable_scope('branch_d2') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 128, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(deconv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.image.resize_images(out, size=(h, w), method=tf.image.ResizeMethod.BILINEAR)
        branch_d2 = tf.nn.sigmoid(out,name='branch_d2')

    # deconv2_1/unpooling
    deconv2_1 = unpool2d(deconv3_2, pool_parameters[-4])

    # deconv2_2
    with tf.variable_scope('deconv2_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 128, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(deconv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        deconv2_2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='deconv2_2')

    # branch_d1
    with tf.variable_scope('branch_d1') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 64, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(deconv2_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.image.resize_images(out, size=(h, w), method=tf.image.ResizeMethod.BILINEAR)
        branch_d1 = tf.nn.sigmoid(out,name='branch_d1')

    # deconv1_1/unpooling
    deconv1_1 = unpool2d(deconv2_2, pool_parameters[-5])

    # deconv1_2
    with tf.variable_scope('deconv1_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(deconv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        deconv1_2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='deconv1_2')


    # branch_d0
    with tf.variable_scope('branch_d0') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 64, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(deconv1_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        #out = tf.image.resize_images(out, size=(h, w), method=tf.image.ResizeMethod.BILINEAR)
        branch_d0 = tf.nn.sigmoid(out,name='branch_d0')


    pred = tf.concat([branch_d5, branch_d4, branch_d3, branch_d2, branch_d1, branch_d0],axis=-1)

    # pred
    with tf.variable_scope('pred_alpha') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 6, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pred, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        pred_alpha = tf.nn.sigmoid(out)

    return branch_e1,branch_e2,branch_e3,branch_e4,branch_e5,branch_d0,branch_d1,branch_d2,branch_d3,branch_d4,branch_d5,pred_alpha,en_parameters


#IOU损失
def loss_IOU(y_pred,y_true):
    H, W, _ = y_pred.get_shape().as_list()[1:]
    flat_logits = tf.reshape(y_pred, [-1, H * W])
    flat_labels = tf.reshape(y_true, [-1, H * W])
    intersection = 2 * tf.reduce_sum(flat_logits * flat_labels, axis=1) + 1e-7
    denominator = tf.reduce_sum(flat_logits, axis=1) + tf.reduce_sum(flat_labels, axis=1) + 1e-7
    iou = 1 - tf.reduce_mean(intersection / denominator)

    return iou

def loss_CE(y_pred,y_true):
    '''flat_logits = tf.reshape(y_pred,[-1,num_class])
    flat_labels = tf.reshape(y_true,[-1,num_class])
    class_weights = tf.constant(loss_weight,dtype=np.float32)
    weight_map = tf.multiply(flat_labels,class_weights)
    weight_map = tf.reduce_sum(weight_map,axis=1)

    loss_map = tf.nn.softmax_cross_entropy_with_logits(labels=flat_labels,logits=flat_logits)

    #weighted_loss = tf.multiply(loss_map,weight_map)

    #cross_entropy_mean = tf.reduce_mean(weighted_loss)
    cross_entropy_mean = -tf.reduce_mean(tf.reduce_sum(y_true*tf.log(y_pred)))'''

    #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    #cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = tf.sqrt(tf.square(y_pred - y_true) + 1e-12)
    cross_entropy_mean = tf.reduce_mean(loss)

    return cross_entropy_mean


def train_op(loss,learning_rate):

    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    return optimizer.minimize(loss,global_step=global_step)


