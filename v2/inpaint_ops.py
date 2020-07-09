import logging
import math

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# from neuralgym.ops.layers import *
# from neuralgym.ops.loss_ops import *
# from neuralgym.ops.gan_ops import *
# from neuralgym.ops.summary_ops import *


logger = logging.getLogger()
np.random.seed(2018)

class Flatten(tf.keras.layers.Layer):
    def __init__(self, name):
        super(Flatten, self).__init__()
        # self.name = name
    def call(self, x, training=True):
        flattened = tf.reshape(x, [tf.shape(x)[0], -1])
        return flattened

class GenConvLayer(tf.keras.layers.Layer):
    """Define conv for generator.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    def __init__(self, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.elu, training=True):
        super(GenConvLayer, self).__init__()
        self.cnum = cnum
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        # self.name = name
        assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
        self.padding = padding
        self.layer_pad_type = padding
        if padding == 'SYMMETRIC' or padding == 'REFELECT':
            self.layer_pad_type = 'VALID'
        self.activation = activation
        self.training = training
        self.conv = tf.keras.layers.Conv2D(filters=self.cnum, kernel_size=self.ksize, strides=self.stride, padding=self.layer_pad_type, name=name)
    # def build(self, input_shape):
    def call(self, inputs, training=True):
        if self.padding == 'SYMMETRIC' or self.padding == 'REFELECT':
            p = int(self.rate * (self.ksize - 1) / 2)
            inputs = tf.pad(tensor=inputs, paddings=[[0,0], [p, p], [p, p], [0,0]], mode=self.padding)
        x = self.conv(inputs)
        if (self.cnum == 3 or self.activation is None):
            # conv for output
            return x
        x, y = tf.split(x, 2, 3)
        x = self.activation(x)
        y = tf.nn.sigmoid(y)
        x = x * y
        return x

class resize(tf.keras.layers.Layer):
    def __init__(self, scale=2, to_shape=None, align_corners=True, dynamic=False, name='resize', func='bilinear'):
        """Resize a given image.
        Originated from https://github.com/JiahuiYu/neuralgym/blob/88292adb524186693a32404c0cfdc790426ea441/neuralgym/ops/layers.py#L141
        """
        super(resize, self).__init__()
        # self.name = name
        self.scale = scale
        self.to_shape = to_shape
        self.align_corners = align_corners
        self.dynamic = dynamic
        if func == 'bilinear':
            self.func = tf.image.ResizeMethod.BILINEAR
        elif func == 'nearest':
            self.func = tf.image.ResizeMethod.NEAREST_NEIGHBOR

    def call(self, inputs, training=True):
        if self.dynamic:
            # NOTE: there seems no dynamic calls in deepfill
            xs = tf.cast(tf.shape(x), tf.float32)
            new_xs = [tf.cast(xs[1]*scale, tf.int32),
                    tf.cast(xs[2]*scale, tf.int32)]
        else:
            xs = x.get_shape().as_list()
            new_xs = [int(xs[1]*scale), int(xs[2]*scale)]
        if self.to_shape is None:
            x = tf.compat.v1.image.resize(images=inputs, size=new_xs, method=self.func, align_corners=self.align_corners)
        else:
            x = tf.compat.v1.image.resize(images=inputs, size=[to_shape[0], to_shape[1]], method=self.func, align_corners=self.align_corners)
        return flattened

# def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False,
#            func=tf.image.resize_bilinear, name='resize'):
#     if dynamic:
#         xs = tf.cast(tf.shape(x), tf.float32)
#         new_xs = [tf.cast(xs[1]*scale, tf.int32),
#                   tf.cast(xs[2]*scale, tf.int32)]
#     else:
#         xs = x.get_shape().as_list()
#         new_xs = [int(xs[1]*scale), int(xs[2]*scale)]
#     with tf.variable_scope(name):
#         if to_shape is None:
#             x = func(x, new_xs, align_corners=align_corners)
#         else:
#             x = func(x, [to_shape[0], to_shape[1]],
#                      align_corners=align_corners)
#     return x

class GenDeconvLayer(tf.keras.layers.Layer):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    def __init__(self, cnum, name='upsample', padding='SAME', training=True):
        super(GenDeconvLayer, self).__init__()
        self.cnum = cnum
        # self.name = name
        self.padding = padding
        self.training = training
        self.conv = GenConvLayer(cnum=cnum, ksize=3, stride=1, name=name + '_conv', padding=self.padding, training=self.training)
    def call(self, inputs, training=True):
        xs = inputs.get_shape().as_list()
        new_xs = [int(xs[1] * 2), int(xs[2] *2)]
        x = tf.compat.v1.image.resize(images=inputs, size=new_xs, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
        x = self.conv(x)
        return x

def l2_norm(input_x, epsilon=1e-12):
    input_x_norm = input_x / (tf.reduce_sum(input_x**2)**0.5 + epsilon)
    return input_x_norm

# def power_iteration(u, ite):
#     v_ = tf.matmul(u, tf.transpose(w_mat))
#     v_hat = l2_norm(v_)
#     u_ = tf.matmul(v_hat, w_mat)
#     u_hat = l2_norm(u_)
#     return u_hat, v_hat, ite+1

class kernel_spectral_norm(tf.keras.layers.Layer):
    def __init__(self, iteration=1, name='kernel_sn'):
        super(kernel_spectral_norm, self).__init__()
        self.iteration = iteration
        # self.name = name
    def call(self, inputs):
        # inputs is kernel
        w_shape = inputs.get_shape().as_list()
        w_mat = tf.reshape(inputs, [-1, w_shape[-1]])
        u = tf.Variable(name='u', shape=[1, w_shape[-1]], initial_value=tf.random.truncated_normal(shape=[1, w_shape[-1]]), trainable=False)

        # u_hat, v_hat,_ = power_iteration(u, self.iteration)
        v_ = tf.matmul(u, tf.transpose(w_mat))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w_mat)
        u_hat = l2_norm(u_)
        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
        w_mat = w_mat / sigma
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_mat, w_shape)
        return w_norm
    

class Conv2DSepctralNorm(tf.keras.layers.Conv2D):
    def build(self, input_shape):
        super(Conv2DSepctralNorm, self).build(input_shape)
        self.sn = kernel_spectral_norm()
        self.kernel = self.sn(self.kernel)

class conv2d_spectral_norm(tf.keras.layers.Layer):
    """
    https://github.com/JiahuiYu/neuralgym/blob/88292adb524186693a32404c0cfdc790426ea441/neuralgym/ops/gan_ops.py#L144
    """
    def __init__(self, filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None):
        super(conv2d_spectral_norm, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.trainable = trainable
        # self.name = name

        self.layer = Conv2DSepctralNorm(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name)

    # def build()
    def call(self, inputs, training=True):
        return self.layer.apply(inputs)


class DisConvLayer(tf.keras.layers.Layer):
    """Define conv for discriminator.
    Activation is set to leaky_relu.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    def __init__(self, cnum, ksize=5, stride=2, name='conv', training=True):
        super(DisConvLayer, self).__init__()
        self.cnum = cnum
        self.ksize = ksize
        self.stride = stride
        # self.name = name
        self.training = training
        self.conv = conv2d_spectral_norm(self.cnum, self.ksize, self.stride, 'SAME', name=name)
    def call(self, inputs, training=True):
        x = self.conv(inputs)
        x = tf.nn.leaky_relu(x)
        return x


def random_bbox(FLAGS):
    """Generate a random tlhw.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = FLAGS['img_shapes']
    img_height = img_shape[0]
    img_width = img_shape[1]
    maxt = img_height - FLAGS['vertical_margin'] - FLAGS['height']
    maxl = img_width - FLAGS['horizontal_margin'] - FLAGS['width']
    t = tf.random.uniform(
        [], minval=FLAGS['vertical_margin'], maxval=maxt, dtype=tf.int32)
    l = tf.random.uniform(
        [], minval=FLAGS['horizontal_margin'], maxval=maxl, dtype=tf.int32)
    h = tf.constant(FLAGS['height'])
    w = tf.constant(FLAGS['width'])
    return (t, l, h, w)


def bbox2mask(FLAGS, bbox, name='mask'):
    """Generate mask tensor from bbox.

    Args:
        bbox: tuple, (top, left, height, width)

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,
             bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
        return mask
    with tf.compat.v1.variable_scope(name), tf.device('/cpu:0'):
        img_shape = FLAGS['img_shapes']
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.compat.v1.py_func(
            npmask,
            [bbox, height, width,
             FLAGS['max_delta_height'], FLAGS['max_delta_width']],
            tf.float32, stateful=False)
        mask.set_shape([1] + [height, width] + [1])
    return mask


def brush_stroke_mask(FLAGS, name='mask'):
    """Generate mask tensor from bbox.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 40
    def generate_mask(H, W):
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        mask = np.reshape(mask, (1, H, W, 1))
        return mask
    with tf.compat.v1.variable_scope(name), tf.device('/cpu:0'):
        img_shape = FLAGS['img_shapes']
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.compat.v1.py_func(
            generate_mask,
            [height, width],
            tf.float32, stateful=True)
        mask.set_shape([1] + [height, width] + [1])
    return mask


def local_patch(x, bbox):
    """Crop local patch according to bbox.

    Args:
        x: input
        bbox: (top, left, height, width)

    Returns:
        tf.Tensor: local patch

    """
    x = tf.image.crop_to_bounding_box(x, bbox[0], bbox[1], bbox[2], bbox[3])
    return x

def resize_mask_like(mask, x):
    """Resize mask like shape of x.

    Args:
        mask: Original mask.
        x: To shape of x.

    Returns:
        tf.Tensor: resized mask

    """
    shape = x.get_shape().as_list()[1:3]
    mask_resize = tf.compat.v1.image.resize(images=mask, size=[shape[0], shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    # mask_resize = resize(
    #     mask, to_shape=,
    #     func=tf.compat.v1.image.resize_nearest_neighbor)
    return mask_resize

def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.

    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.

    Returns:
        tf.Tensor: output

    """
    # get shapes
    raw_fs = tf.shape(input=f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # print('>> fs', raw_int_fs)
    # print('>> bs', raw_int_bs)
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.image.extract_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(a=raw_w, perm=[0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.

    # resize f
    scale_rate = 1./rate
    xs = f.get_shape().as_list()
    new_xs = [int(xs[1] * scale_rate), int(xs[2] * scale_rate)]
    # print(new_xs, rate)
    # print([int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)])
    # print(tf.shape(f), tf.shape(b), tf.shape(mask))
    f = tf.compat.v1.image.resize(images=f, size=new_xs, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    # f = resize(f, scale=1./rate, func=tf.compat.v1.image.resize_nearest_neighbor)
    # resize b
    b = tf.compat.v1.image.resize(images=b, size=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    # b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.compat.v1.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        # resize mask
        mask = tf.compat.v1.image.resize(images=mask, size=new_xs, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
        # mask = resize(mask, scale=1./rate, func=tf.compat.v1.image.resize_nearest_neighbor)
    # print(tf.shape(f), tf.shape(b), tf.shape(mask))
    # print(f.shape, b.shape, mask.shape)
    fs = tf.shape(input=f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(input=b)
    int_bs = b.get_shape().as_list()
    w = tf.image.extract_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(a=w, perm=[0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.image.extract_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(a=m, perm=[0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(input_tensor=m, axis=[0,1,2], keepdims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(input_tensor=tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(input=xi, filters=wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(input=yi, filters=fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(a=yi, perm=[0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(input=yi, filters=fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(a=yi, perm=[0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask

        offset = tf.argmax(input=yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        # resize flow
        # flow = resize(flow, scale=rate, func=tf.compat.v1.image.resize_bilinear)
        xs = flow.get_shape().as_list()
        new_xs = [int(xs[1] * scale_rate), int(xs[2] * scale_rate)]
        flow = tf.compat.v1.image.resize(images=flow, size=new_xs, method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    return y, flow


def test_contextual_attention(args):
    """Test contextual attention layer with 3-channel image input
    (instead of n-channel feature).

    """
    import cv2
    import os
    # run on cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    rate = 2
    stride = 1
    grid = rate*stride

    b = cv2.imread(args.imageA)
    b = cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    h, w, _ = b.shape
    b = b[:h//grid*grid, :w//grid*grid, :]
    b = np.expand_dims(b, 0)
    # print(b.shape)
    logger.info('Size of imageA: {}'.format(b.shape))

    f = cv2.imread(args.imageB)
    f = cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    h, w, _ = f.shape
    f = f[:h//grid*grid, :w//grid*grid, :]
    f = np.expand_dims(f, 0)
    # print(f.shape)
    logger.info('Size of imageB: {}'.format(f.shape))

    with tf.compat.v1.Session() as sess:
        bt = tf.constant(b, dtype=tf.float32)
        ft = tf.constant(f, dtype=tf.float32)

        yt, flow = contextual_attention(
            ft, bt, stride=stride, rate=rate,
            training=False, fuse=False)
        y = sess.run(yt)
        cv2.imwrite(args.imageOut, y[0])

def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


COLORWHEEL = make_color_wheel()


def compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img



def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    with tf.compat.v1.variable_scope(name), tf.device('/cpu:0'):
        img = tf.compat.v1.py_func(flow_to_image, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img


def highlight_flow(flow):
    """Convert flow into middlebury color code image.
    """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h,w]
                vi = v[h,w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def highlight_flow_tf(flow, name='flow_to_image'):
    """Tensorflow ops for highlight flow.
    """
    with tf.compat.v1.variable_scope(name), tf.device('/cpu:0'):
        img = tf.compat.v1.py_func(highlight_flow, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img


def image2edge(image):
    """Convert image to edges.
    """
    out = []
    for i in range(image.shape[0]):
        img = cv2.Laplacian(image[i, :, :, :], cv2.CV_64F, ksize=3, scale=2)
        out.append(img)
    return np.float32(np.uint8(out))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageA', default='', type=str, help='Image A as background patches to reconstruct image B.')
    parser.add_argument('--imageB', default='', type=str, help='Image B is reconstructed with image A.')
    parser.add_argument('--imageOut', default='result.png', type=str, help='Image B is reconstructed with image A.')
    args = parser.parse_args()
    test_contextual_attention(args)

    # python inpaint_ops.py --imageA ../examples/places2/case1_input.png --imageB ../examples/places2/case1_input.png
