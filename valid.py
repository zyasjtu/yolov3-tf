from nwk.yolov3 import yolov3
import tensorflow as tf
import numpy as np

def main():
    img = tf.placeholder(tf.float32, [1,256,256,3])
    tng = tf.placeholder('bool', [])
    yolo = yolov3(1, [1,2,3,4,5,6,7,8,9])
    ft_map = yolo.inference(img, tng)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ipt = np.ones([256, 256, 3], np.uint8)
        ft_opt = sess.run( ft_map, feed_dict={img:np.reshape(ipt,(1,256,256,3)), tng:True} )
        print(ft_opt[0].shape, ft_opt[1].shape, ft_opt[2].shape)

if __name__ == '__main__':
    main()        
