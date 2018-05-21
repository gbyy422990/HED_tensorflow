# coding:utf-8
# Bin GAO

import os
import cv2
import glob
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rgb',
                    type=str,
                    default='./new_pig1')
parser.add_argument('--model_dir',
                    type=str,
                    default='./model_base')
parser.add_argument('--pig_save_dir',
                    type=str,
                    default='./result')
parser.add_argument('--pig_save_name',
                    type=str,
                    default='333.jpg')
parser.add_argument('--gpu',
                    type=int,
                    default=1)
flags = parser.parse_args()

h=1024
w=1024

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_model():
    file_meta = os.path.join(flags.model_dir, 'model.ckpt.meta')
    file_ckpt = os.path.join(flags.model_dir, 'model.ckpt')

    saver = tf.train.import_meta_graph(file_meta)
    # tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

    sess = tf.InteractiveSession()
    saver.restore(sess, file_ckpt)
    # print(sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")))
    return sess


def read_image(image_path, gray=False):
    if gray:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main(flags):
    sess = load_model()
    X, mode = tf.get_collection('inputs')
    pred = tf.get_collection('upscore_fuse')[0]

    img_list = os.listdir(flags.rgb)
    for i in img_list:
        if i != '.DS_Store':
            img = os.listdir(flags.rgb+'/'+i)
            for j in img:
                if j != '.DS_Store':
                    #print(j)
                    image = read_image(flags.rgb + '/' + i + '/' + j)
                    origin_shape = image.shape
                    print(origin_shape[:2])
                    image=cv2.resize(image,(h,w))
                    # sess=tf.InteractiveSession()
                    #img = image.reshape((1, 352, 352, 3))

                    label_pred = sess.run(pred, feed_dict={X: np.expand_dims(image,0), mode: False})
                    #print(label_pred)
                    #result = np.reshape(img,(h/2,w/2,2))
                    #result = image[:,:,0]
                    merged = np.squeeze(label_pred)*255
                    #_,merged = cv2.threshold(merged,200,255,cv2.THRESH_BINARY)

                    try:
                        os.mkdir(flags.pig_save_dir + '/' + i)
                    except Exception as e:
                        print(e)


                    #save_name = os.path.join(flags.pig_save_dir + '/' + i, j)

                    save_name = os.path.join(flags.pig_save_dir + '/' + i,j)
                    cv2.imwrite(save_name,merged)




if __name__ == '__main__':
    main(flags)






