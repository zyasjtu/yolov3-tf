import darknet53
from darknet53 import conv
import tensorflow as tf
import numpy as np

class yolov3():

    def __init__(self, cls_nb, acr, img_sz):
        self.cls_nb = cls_nb
        self.acr = acr
        self.img_sz = img_sz

    def conv_block(self, ipt, ft_nb, tng):
        cvlt = ipt
        for i in range(2):
            cvlt = conv(cvlt, ft_nb/2, 1, 1, tng=tng)
            cvlt = conv(cvlt, ft_nb, 3, 1, tng=tng)
        route = conv(cvlt, ft_nb/2, 1, 1, tng=tng)
        opt = conv(route, ft_nb, 3, 1, tng=tng)
        return route, opt

    def detect_layer(self, ipt, acr):
        acr_nb = len(acr)
        opt = conv(ipt, acr_nb*(self.cls_nb+1+4), 1, 1, bn=False, atvt=None)
        return opt

    def inference(self, ipt, tng):
        scale3, scale2, scale1 = darknet53.inference(ipt, tng)
       
        route1, ft_map_1 = self.conv_block(scale1, 1024, tng=tng)
        rst1 = self.detect_layer(ft_map_1, self.acr[6:9])

        usp2 = tf.image.resize_nearest_neighbor(conv(route1,256,1,1,tng=tng), route1.get_shape().as_list()[1:3]*np.array([2,2]))
        cct2 = tf.concat([usp2, scale2], axis=3)

        route2, ft_map_2 = self.conv_block(cct2, 512, tng=tng)
        rst2 = self.detect_layer(ft_map_2, self.acr[3:6])
        
        usp3 = tf.image.resize_nearest_neighbor(conv(route2,128,1,1,tng=tng), route2.get_shape().as_list()[1:3]*np.array([2,2]))
        cct3 = tf.concat([usp3, scale3], axis=3)
        _, ft_map_3 = self.conv_block(cct3, 256, tng=tng)
        rst3 = self.detect_layer(ft_map_3, self.acr[0:3])
        return rst1, rst2, rst3

    def transfer(self, rst, acr):
        grid_sz = rst.shape.as_list()[1:3]
        std = tf.cast(self.img_sz // grid_sz, tf.float32)

        rst = tf.reshape(rst, [-1, grid_sz[0], grid_sz[1], len(acr), 4+1+self.cls_nb])
        ctr, sz, conf, prob = tf.split(rst, [2,2,1,self.cls_nb], axis=-1)
        print(ctr.shape, sz.shape, conf.shape, prob.shape)

        grid_x = tf.range(grid_sz[0], dtype=tf.int32)
        grid_y = tf.range(grid_sz[1], dtype=tf.int32)
        x, y = tf.meshgrid(grid_x, grid_y)
        x_ost = tf.reshape(x, (-1, 1))
        y_ost = tf.reshape(y, (-1, 1))
        ost = tf.concat([x_ost, y_ost], axis=-1)
        ost = tf.reshape(ost, [grid_sz[0], grid_sz[1], 1, 2])
        ost = tf.cast(ost, tf.float32)

        ctr = (tf.nn.sigmoid(ctr) + ost) * std
        sz = tf.exp(sz) * acr
        box = tf.reshape(tf.concat([ctr, sz], axis=-1), [-1, grid_sz[0]*grid_sz[1]*3, 4])
        conf = tf.reshape(conf, [-1, grid_sz[0]*grid_sz[1]*3, 1])
        prob = tf.reshape(prob, [-1, grid_sz[0]*grid_sz[1]*3, self.cls_nb])

        return box, tf.sigmoid(conf), tf.sigmoid(prob)

    def predict(self, rst1, rst2, rst3):
        # rst1 --> [None, 13, 13, 255]
        box1, conf1, prob1 = self.transfer(rst1, self.acr[6:9])
        # rst2 --> [None, 26, 26, 255]
        box2, conf2, prob2 = self.transfer(rst2, self.acr[3:6])
        # rst1 --> [None, 52, 52, 255]
        box3, conf3, prob3 = self.transfer(rst3, self.acr[0:3])

        box = tf.concat([box1, box2, box3], axis=1)
        conf = tf.concat([conf1, conf2, conf3], axis=1)
        prob = tf.concat([prob1, prob2, prob3], axis=1)

        return box, conf, prob
