""" common model for DCGAN """
import logging

import cv2
# import neuralgym as ng
import tensorflow as tf
# from tensorflow.contrib.framework.python.ops import arg_scope

# from neuralgym.models import Model

from inpaint_ops import GenConvLayer, GenDeconvLayer, DisConvLayer, Flatten
from inpaint_ops import random_bbox, bbox2mask, local_patch, brush_stroke_mask
from inpaint_ops import resize_mask_like, contextual_attention


logger = logging.getLogger()


def gan_hinge_loss(pos, neg, value=1., name='gan_hinge_loss'):
    """
    gan with hinge loss:
    https://github.com/pfnet-research/sngan_projection/blob/c26cedf7384c9776bcbe5764cb5ca5376e762007/updater.py
    """
    with tf.compat.v1.variable_scope(name):
        hinge_pos = tf.reduce_mean(tf.nn.relu(1-pos))
        hinge_neg = tf.reduce_mean(tf.nn.relu(1+neg))
        d_loss = tf.add(.5 * hinge_pos, .5 * hinge_neg)
        g_loss = -tf.reduce_mean(neg)
    return g_loss, d_loss


class InpaintGenerator(tf.keras.Model):
    def __init__(self, reuse=False, training=True, padding='SAME', name='inpaint_net'):
        super(InpaintGenerator, self).__init__(name=name)

        cnum = 48
        # stage 1
        self.s1_1 = tf.keras.Sequential([
            GenConvLayer(cnum, 5, 1, name=name + '_conv1'),
            GenConvLayer(2 * cnum, 3, 2, name=name + '_conv2_downsample'),
            GenConvLayer(2 * cnum, 3, 1, name=name + '_conv3'),
            GenConvLayer(4 * cnum, 3, 2, name=name + '_conv4_downsample'),
            GenConvLayer(4 * cnum, 3, 1, name=name + '_conv5'),
            GenConvLayer(4 * cnum, 3, 1, name=name + '_conv6')
        ])

        self.s1_2 = tf.keras.Sequential([
            GenConvLayer(4*cnum, 3, rate=2, name=name + '_conv7_atrous'),
            GenConvLayer(4*cnum, 3, rate=4, name=name + '_conv8_atrous'),
            GenConvLayer(4*cnum, 3, rate=8, name=name + '_conv9_atrous'),
            GenConvLayer(4*cnum, 3, rate=16, name=name + '_conv10_atrous'),
            GenConvLayer(4*cnum, 3, 1, name=name + '_conv11'),
            GenConvLayer(4*cnum, 3, 1, name=name + '_conv12'),
            GenDeconvLayer(2*cnum, name=name + '_conv13_upsample'),
            GenConvLayer(2*cnum, 3, 1, name=name + '_conv14'),
            GenDeconvLayer(cnum, name=name + '_conv15_upsample'),
            GenConvLayer(cnum//2, 3, 1, name=name + '_conv16'),
            GenConvLayer(3, 3, 1, activation=None, name=name + '_conv17')
        ])

        self.s2 = tf.keras.Sequential([
            GenConvLayer(cnum, 5, 1, name=name + '_xconv1'),
            GenConvLayer(cnum, 3, 2, name=name + '_xconv2_downsample'),
            GenConvLayer(2*cnum, 3, 1, name=name + '_xconv3'),
            GenConvLayer(2*cnum, 3, 2, name=name + '_xconv4_downsample'),
            GenConvLayer(4*cnum, 3, 1, name=name + '_xconv5'),
            GenConvLayer(4*cnum, 3, 1, name=name + '_xconv6'),
            GenConvLayer(4*cnum, 3, rate=2, name=name + '_xconv7_atrous'),
            GenConvLayer(4*cnum, 3, rate=4, name=name + '_xconv8_atrous'),
            GenConvLayer(4*cnum, 3, rate=8, name=name + '_xconv9_atrous'),
            GenConvLayer(4*cnum, 3, rate=16, name=name + '_xconv10_atrous')
        ])

        # attention branch
        self.attn = tf.keras.Sequential([
            GenConvLayer(cnum, 5, 1, name=name + '_pmconv1'),
            GenConvLayer(cnum, 3, 2, name=name + '_pmconv2_downsample'),
            GenConvLayer(2 * cnum, 3, 1, name=name + '_pmconv3'),
            GenConvLayer(4 * cnum, 3, 2, name=name + '_pmconv4_downsample'),
            GenConvLayer(4 * cnum, 3, 1, name=name + '_pmconv5'),
            GenConvLayer(4 * cnum, 3, 1, name=name +
                         '_pmconv6', activation=tf.nn.relu),
        ])

        self.attn_2 = tf.keras.Sequential([
            GenConvLayer(4 * cnum, 3, 1, name=name + '_pmconv9'),
            GenConvLayer(4 * cnum, 3, 1, name=name + '_pmconv10')
        ])

        self.final = tf.keras.Sequential([
            GenConvLayer(4*cnum, 3, 1, name=name + '_allconv11'),
            GenConvLayer(4*cnum, 3, 1, name=name + '_allconv12'),
            GenDeconvLayer(2*cnum, name=name + '_allconv13_upsample'),
            GenConvLayer(2*cnum, 3, 1, name=name + '_allconv14'),
            GenDeconvLayer(cnum, name=name + '_allconv15_upsample'),
            GenConvLayer(cnum//2, 3, 1, name=name + '_allconv16'),
            GenConvLayer(3, 3, 1, activation=None, name=name + '_allconv17')
        ])

    def call(self, x, mask, training):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        xin = x
        offset_flow = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x, ones_x * mask], axis=3)

        cnum = 48

        # stage1
        x = self.s1_1(x)
        mask_s = resize_mask_like(mask, x)
        x = self.s1_2(x)
        x = tf.keras.activations.tanh(x)

        x_stage1 = x

        # stage2
        # paste result as input
        x = x * mask + xin[:, :, :, 0:3] * (1. - mask)
        x.set_shape(xin[:, :, :, 0:3].get_shape().as_list())

        # conv branch
        xnow = x
        x = self.s2(xnow)
        x_hallu = x

        # attention branch
        x = self.attn(xnow)
        x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
        x = self.attn_2(x)
        pm = x
        x = tf.concat([x_hallu, pm], axis=3)

        # final part
        x_stage2 = self.final(x)
        x_stage2 = tf.keras.activations.tanh(x_stage2)

        return x_stage1, x_stage2, offset_flow


class InpaintDiscriminator(tf.keras.Model):
    def __init__(self, reuse=False, training=True, name='inpaint_discriminator'):
        super(InpaintDiscriminator, self).__init__(name=name)
        self.reuse = reuse
        self.training = training
        cnum = 64
        self.net = tf.keras.Sequential([
            DisConvLayer(cnum, name=name + '_conv1', training=training),
            DisConvLayer(cnum*2, name=name + '_conv2', training=training),
            DisConvLayer(cnum*4, name=name + '_conv3', training=training),
            DisConvLayer(cnum*4, name=name + '_conv4', training=training),
            DisConvLayer(cnum*4, name=name + '_conv5', training=training),
            DisConvLayer(cnum*4, name=name + '_conv6', training=training),
            Flatten(name=name + '_flatten')
        ])

    def call(self, x, training):
        return self.net(x)
