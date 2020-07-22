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
        # scalar_summary('pos_hinge_avg', hinge_pos)
        # scalar_summary('neg_hinge_avg', hinge_neg)
        d_loss = tf.add(.5 * hinge_pos, .5 * hinge_neg)
        g_loss = -tf.reduce_mean(neg)
        # scalar_summary('d_loss', d_loss)
        # scalar_summary('g_loss', g_loss)
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
            GenConvLayer(4 * cnum, 3, 1, name=name + '_pmconv6', activation=tf.nn.relu),
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


# class InpaintCAModel(Model):
#     def __init__(self):
#         super().__init__('InpaintCAModel')

#     def build_inpaint_net(self, x, mask, reuse=False,
#                           training=True, padding='SAME', name='inpaint_net'):
#         """Inpaint network.

#         Args:
#             x: incomplete image, [-1, 1]
#             mask: mask region {0, 1}
#         Returns:
#             [-1, 1] as predicted image
#         """
#         xin = x
#         offset_flow = None
#         ones_x = tf.ones_like(x)[:, :, :, 0:1]
#         x = tf.concat([x, ones_x, ones_x*mask], axis=3)

#         # two stage network
#         cnum = 48
#         with tf.compat.v1.variable_scope(name, reuse=reuse), \
#                 arg_scope([gen_conv, gen_deconv],
#                           training=training, padding=padding):
#             # stage1
#             x = gen_conv(x, cnum, 5, 1, name='conv1')
#             x = gen_conv(x, 2*cnum, 3, 2, name='conv2_downsample')
#             x = gen_conv(x, 2*cnum, 3, 1, name='conv3')
#             x = gen_conv(x, 4*cnum, 3, 2, name='conv4_downsample')
#             x = gen_conv(x, 4*cnum, 3, 1, name='conv5')
#             x = gen_conv(x, 4*cnum, 3, 1, name='conv6')
#             mask_s = resize_mask_like(mask, x)
#             x = gen_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous')
#             x = gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous')
#             x = gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous')
#             x = gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous')
#             x = gen_conv(x, 4*cnum, 3, 1, name='conv11')
#             x = gen_conv(x, 4*cnum, 3, 1, name='conv12')
#             x = gen_deconv(x, 2*cnum, name='conv13_upsample')
#             x = gen_conv(x, 2*cnum, 3, 1, name='conv14')
#             x = gen_deconv(x, cnum, name='conv15_upsample')
#             x = gen_conv(x, cnum//2, 3, 1, name='conv16')
#             x = gen_conv(x, 3, 3, 1, activation=None, name='conv17')
#             x = tf.nn.tanh(x)
#             x_stage1 = x

#             # stage2, paste result as input
#             x = x*mask + xin[:, :, :, 0:3]*(1.-mask)
#             x.set_shape(xin[:, :, :, 0:3].get_shape().as_list())
#             # conv branch
#             # xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
#             xnow = x
#             x = gen_conv(xnow, cnum, 5, 1, name='xconv1')
#             x = gen_conv(x, cnum, 3, 2, name='xconv2_downsample')
#             x = gen_conv(x, 2*cnum, 3, 1, name='xconv3')
#             x = gen_conv(x, 2*cnum, 3, 2, name='xconv4_downsample')
#             x = gen_conv(x, 4*cnum, 3, 1, name='xconv5')
#             x = gen_conv(x, 4*cnum, 3, 1, name='xconv6')
#             x = gen_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous')
#             x = gen_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous')
#             x = gen_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous')
#             x = gen_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous')
#             x_hallu = x
#             # attention branch
#             x = gen_conv(xnow, cnum, 5, 1, name='pmconv1')
#             x = gen_conv(x, cnum, 3, 2, name='pmconv2_downsample')
#             x = gen_conv(x, 2*cnum, 3, 1, name='pmconv3')
#             x = gen_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample')
#             x = gen_conv(x, 4*cnum, 3, 1, name='pmconv5')
#             x = gen_conv(x, 4*cnum, 3, 1, name='pmconv6',
#                                 activation=tf.nn.relu)
#             x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
#             x = gen_conv(x, 4*cnum, 3, 1, name='pmconv9')
#             x = gen_conv(x, 4*cnum, 3, 1, name='pmconv10')
#             pm = x
#             x = tf.concat([x_hallu, pm], axis=3)

#             x = gen_conv(x, 4*cnum, 3, 1, name='allconv11')
#             x = gen_conv(x, 4*cnum, 3, 1, name='allconv12')
#             x = gen_deconv(x, 2*cnum, name='allconv13_upsample')
#             x = gen_conv(x, 2*cnum, 3, 1, name='allconv14')
#             x = gen_deconv(x, cnum, name='allconv15_upsample')
#             x = gen_conv(x, cnum//2, 3, 1, name='allconv16')
#             x = gen_conv(x, 3, 3, 1, activation=None, name='allconv17')
#             x = tf.nn.tanh(x)
#             x_stage2 = x
#         return x_stage1, x_stage2, offset_flow

#     def build_sn_patch_gan_discriminator(self, x, reuse=False, training=True):
#         with tf.compat.v1.variable_scope('sn_patch_gan', reuse=reuse):
#             cnum = 64
#             x = dis_conv(x, cnum, name='conv1', training=training)
#             x = dis_conv(x, cnum*2, name='conv2', training=training)
#             x = dis_conv(x, cnum*4, name='conv3', training=training)
#             x = dis_conv(x, cnum*4, name='conv4', training=training)
#             x = dis_conv(x, cnum*4, name='conv5', training=training)
#             x = dis_conv(x, cnum*4, name='conv6', training=training)
#             x = Flatten(x, name='flatten')
#             return x

#     def build_gan_discriminator(
#             self, batch, reuse=False, training=True):
#         with tf.compat.v1.variable_scope('discriminator', reuse=reuse):
#             d = self.build_sn_patch_gan_discriminator(
#                 batch, reuse=reuse, training=training)
#             return d

#     def build_graph_with_losses(
#             self, FLAGS, batch_data, training=True, summary=False,
#             reuse=False):
#         if FLAGS.guided:
#             batch_data, edge = batch_data
#             edge = edge[:, :, :, 0:1] / 255.
#             edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)
#         batch_pos = batch_data / 127.5 - 1.
#         # generate mask, 1 represents masked point
#         bbox = random_bbox(FLAGS)
#         regular_mask = bbox2mask(FLAGS, bbox, name='mask_c')
#         irregular_mask = brush_stroke_mask(FLAGS, name='mask_c')
#         mask = tf.cast(
#             tf.logical_or(
#                 tf.cast(irregular_mask, tf.bool),
#                 tf.cast(regular_mask, tf.bool),
#             ),
#             tf.float32
#         )

#         batch_incomplete = batch_pos*(1.-mask)
#         if FLAGS.guided:
#             edge = edge * mask
#             xin = tf.concat([batch_incomplete, edge], axis=3)
#         else:
#             xin = batch_incomplete
#         x1, x2, offset_flow = self.build_inpaint_net(
#             xin, mask, reuse=reuse, training=training,
#             padding=FLAGS.padding)
#         batch_predicted = x2
#         losses = {}
#         # apply mask and complete image
#         batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
#         # local patches
#         losses['ae_loss'] = FLAGS.l1_loss_alpha * tf.reduce_mean(input_tensor=tf.abs(batch_pos - x1))
#         losses['ae_loss'] += FLAGS.l1_loss_alpha * tf.reduce_mean(input_tensor=tf.abs(batch_pos - x2))
#         # if summary:
#         #     scalar_summary('losses/ae_loss', losses['ae_loss'])
#         #     if FLAGS.guided:
#         #         viz_img = [
#         #             batch_pos,
#         #             batch_incomplete + edge,
#         #             batch_complete]
#         #     else:
#         #         viz_img = [batch_pos, batch_incomplete, batch_complete]
#         #     if offset_flow is not None:
#         #         viz_img.append(
#         #             resize(offset_flow, scale=4,
#         #                    func=tf.image.resize_bilinear))
#         #     (
#         #         tf.concat(viz_img, axis=2),
#         #         'raw_incomplete_predicted_complete', FLAGS.viz_max_out)

#         # gan
#         batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
#         if FLAGS.gan_with_mask:
#             batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [FLAGS.batch_size*2, 1, 1, 1])], axis=3)
#         if FLAGS.guided:
#             # conditional GANs
#             batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(edge, [2, 1, 1, 1])], axis=3)
#         # wgan with gradient penalty
#         if FLAGS.gan == 'sngan':
#             pos_neg = self.build_gan_discriminator(batch_pos_neg, training=training, reuse=reuse)
#             pos, neg = tf.split(pos_neg, 2)
#             g_loss, d_loss = gan_hinge_loss(pos, neg)
#             losses['g_loss'] = g_loss
#             losses['d_loss'] = d_loss
#         else:
#             raise NotImplementedError('{} not implemented.'.format(FLAGS.gan))
#         # if summary:
#         #     # summary the magnitude of gradients from different losses w.r.t. predicted image
#         #     gradients_summary(losses['g_loss'], batch_predicted, name='g_loss')
#         #     gradients_summary(losses['g_loss'], x2, name='g_loss_to_x2')
#         #     # gradients_summary(losses['ae_loss'], x1, name='ae_loss_to_x1')
#         #     gradients_summary(losses['ae_loss'], x2, name='ae_loss_to_x2')
#         losses['g_loss'] = FLAGS.gan_loss_alpha * losses['g_loss']
#         if FLAGS.ae_loss:
#             losses['g_loss'] += losses['ae_loss']
#         g_vars = tf.compat.v1.get_collection(
#             tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
#         d_vars = tf.compat.v1.get_collection(
#             tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
#         return g_vars, d_vars, losses

#     def build_infer_graph(self, FLAGS, batch_data, bbox=None, name='val'):
#         """
#         """
#         if FLAGS.guided:
#             batch_data, edge = batch_data
#             edge = edge[:, :, :, 0:1] / 255.
#             edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)
#         regular_mask = bbox2mask(FLAGS, bbox, name='mask_c')
#         irregular_mask = brush_stroke_mask(FLAGS, name='mask_c')
#         mask = tf.cast(
#             tf.logical_or(
#                 tf.cast(irregular_mask, tf.bool),
#                 tf.cast(regular_mask, tf.bool),
#             ),
#             tf.float32
#         )

#         batch_pos = batch_data / 127.5 - 1.
#         batch_incomplete = batch_pos*(1.-mask)
#         if FLAGS.guided:
#             # False
#             edge = edge * mask
#             xin = tf.concat([batch_incomplete, edge], axis=3)
#         else:
#             xin = batch_incomplete
#         # inpaint
#         x1, x2, offset_flow = self.build_inpaint_net(
#             xin, mask, reuse=True,
#             training=False, padding=FLAGS.padding)
#         batch_predicted = x2
#         # apply mask and reconstruct
#         batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
#         # global image visualization
#         # if FLAGS.guided:
#         #     viz_img = [
#         #         batch_pos,
#         #         batch_incomplete + edge,
#         #         batch_complete]
#         # else:
#         #     viz_img = [batch_pos, batch_incomplete, batch_complete]
#         # if offset_flow is not None:
#         #     viz_img.append(
#         #         resize(offset_flow, scale=4,
#         #                func=tf.compat.v1.image.resize_bilinear))
#         # (
#         #     tf.concat(viz_img, axis=2),
#         #     name+'_raw_incomplete_complete', FLAGS.viz_max_out)
#         return batch_complete

#     def build_static_infer_graph(self, FLAGS, batch_data, name):
#         """
#         """
#         # generate mask, 1 represents masked point
#         bbox = (tf.constant(FLAGS.height//2), tf.constant(FLAGS.width//2),
#                 tf.constant(FLAGS.height), tf.constant(FLAGS.width))
#         return self.build_infer_graph(FLAGS, batch_data, bbox, name)


#     def build_server_graph(self, FLAGS, batch_data, reuse=False, is_training=False):
#         """
#         """
#         # generate mask, 1 represents masked point
#         if FLAGS.guided:
#             batch_raw, edge, masks_raw = tf.split(batch_data, 3, axis=2)
#             edge = edge[:, :, :, 0:1] / 255.
#             edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)
#         else:
#             batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
#         masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

#         batch_pos = batch_raw / 127.5 - 1.
#         batch_incomplete = batch_pos * (1. - masks)
#         if FLAGS.guided:
#             edge = edge * masks[:, :, :, 0:1]
#             xin = tf.concat([batch_incomplete, edge], axis=3)
#         else:
#             xin = batch_incomplete
#         # inpaint
#         x1, x2, flow = self.build_inpaint_net(
#             xin, masks, reuse=reuse, training=is_training)
#         batch_predict = x2
#         # apply mask and reconstruct
#         batch_complete = batch_predict*masks + batch_incomplete*(1-masks)
#         return batch_complete
