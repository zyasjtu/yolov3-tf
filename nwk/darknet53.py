import tensorflow as tf


def conv(ipt, ft_nb, kn_sz, std, bn=True, tng=True, atvt=tf.nn.leaky_relu):
    opt = tf.layers.conv2d(inputs=ipt, filters=ft_nb, kernel_size=kn_sz, strides=(std,std), padding='same')
    if bn:
        opt = tf.layers.batch_normalization(opt, training=tng)
    if not (atvt is None):
        opt = atvt(opt)
    return opt

def conv_block(ipt, ft_nb, tng=True):
    conv1 = conv(ipt, ft_nb/2, 1, 1, tng=tng)
    conv2 = conv(conv1, ft_nb, 3, 1, tng=tng)
    opt = ipt + conv2
    return opt

def inference(ipt, tng):
    cvlt = conv(ipt, 32, 3, 1, tng=tng)
    dsp1 = conv(cvlt, 64, 3, 2, tng=tng)

    c_b_1 = conv_block(dsp1, 64, tng=tng)
    dsp2 = conv(c_b_1, 128, 3, 2, tng=tng)
    
    c_b_2 = dsp2
    for i in range(2):
        c_b_2 = conv_block(c_b_2, 128, tng=tng)
    dsp3 = conv(c_b_2, 256, 3, 2, tng=tng)

    c_b_3 = dsp3
    for i in range(8):
        c_b_3 = conv_block(c_b_3, 256, tng=tng)
    dsp4 = conv(c_b_3, 512, 3, 2, tng=tng)

    c_b_4 = dsp4
    for i in range(8):
        c_b_4 = conv_block(c_b_4, 512, tng=tng)
    dsp5 = conv(c_b_4, 1024, 3, 2, tng=tng)

    c_b_5 = dsp5
    for i in range(4):
        c_b_5 = conv_block(c_b_5, 1024, tng=tng)

    return c_b_3, c_b_4, c_b_5
