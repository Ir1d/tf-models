import os
import glob
import tqdm
import yaml
import datetime

import tensorflow as tf
import numpy as np

from inpaint_model import InpaintGenerator, InpaintDiscriminator, gan_hinge_loss
from inpaint_ops import random_bbox, bbox2mask, brush_stroke_mask


def prepare_for_training(ds, shuffle_buffer_size=1000, batch_size=1, repeat=False):
    # https://www.tensorflow.org/tutorials/load_data/images
    # places2 toooooo big cant cache
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # print(repeat)
    if repeat:
        # Repeat forever for train
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=True)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def decode_png(img):
    # convert the compressed string to a 3D uint8 tensor
    # use decode_png instead of decode_image https://stackoverflow.com/a/49101717/4597306
    img = tf.image.decode_png(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # dont resize here, leave resize in the train loop to generate patches on-the-fly
    return img


def decode_jpeg(img):
    # convert the compressed string to a 3D uint8 tensor
    # use decode_png instead of decode_image https://stackoverflow.com/a/49101717/4597306
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # dont resize here, leave resize in the train loop to generate patches on-the-fly
    return img


def process_path(file_path, resize=True, val=False):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_png(img)
    if resize:
        img = tf.image.resize(img, size=[256, 256])
    return img


def process_val_pair(file_path, resize=True):
    # load the raw data from the file as a string
    gt_img = process_path(file_path, resize=True, val=True)
    # read input image
    input_path = tf.strings.regex_replace(
        file_path, '_output.png', '_input.png')
    input_img = process_path(input_path, resize=True, val=True)
    # read mask image
    mask_path = tf.strings.regex_replace(file_path, '_output.png', '_mask.png')
    mask_img = process_path(mask_path, resize=True, val=True)
    mask_img = tf.image.rgb_to_grayscale(mask_img)
    return input_img, gt_img, mask_img


def get_train_iter(bs=1):
    data_dir = '/home/ir1d/data_large/'
    data_dir = '/data/datasets/places2/train/'
    list_ds = tf.data.Dataset.list_files(str(data_dir + '*.png'))
    new_ds = list_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = prepare_for_training(new_ds, batch_size=bs, repeat=True)
    return iter(train_ds)


def get_val_ds(bs=1):
    data_dir = 'examples/places2/'
    list_ds = tf.data.Dataset.list_files(str(data_dir + '*_output.png'))
    new_ds = list_ds.map(
        process_val_pair, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = prepare_for_training(new_ds, batch_size=bs, repeat=False)
    num_elements = tf.data.experimental.cardinality(val_ds).numpy()
    print('valset size:', num_elements)
    return (val_ds), num_elements


if __name__ == "__main__":
    FLAGS = yaml.load(open('inpaint.yml', 'r'), Loader=yaml.FullLoader)
    img_shapes = FLAGS['img_shapes']  # 256x256
    G = InpaintGenerator()
    D = InpaintDiscriminator()
    lr = tf.Variable(name='lr', initial_value=1e-4, trainable=False, shape=[])
    d_optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.5, beta_2=0.999)
    g_optimizer = d_optimizer

    G.load_weights('logs/20200729-160205/models/G')
    D.load_weights('logs/20200729-160205/models/D')

    # data
    train_iter = get_train_iter(bs=FLAGS['batch_size'])
    val_ds, val_size = get_val_ds()
    best_psnr = tf.constant(0.0)
    log_folder = 'logs/'
    os.makedirs(log_folder, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(log_folder, current_time, 'train')
    model_dir = os.path.join(log_folder, current_time, 'models')
    os.makedirs(model_dir, exist_ok=True)
    print('Logging to ', train_log_dir)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    @tf.function
    def train_step(iter_idx):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            batch_pos = next(train_iter)
            bbox = random_bbox(FLAGS)
            regular_mask = bbox2mask(FLAGS, bbox, name='mask_c')
            irregular_mask = brush_stroke_mask(FLAGS, name='mask_c')
            mask = tf.cast(
                tf.logical_or(
                    tf.cast(irregular_mask, tf.bool),
                    tf.cast(regular_mask, tf.bool),
                ),
                tf.float32
            )
            # mask is 0-1
            batch_incomplete = batch_pos*(1.-mask)
            xin = batch_incomplete
            x1, x2, offset_flow = G(xin, mask, training=True)
            batch_predicted = x2

            losses = {}
            batch_complete = batch_predicted * \
                mask + batch_incomplete * (1. - mask)
            losses['ae_loss'] = FLAGS['l1_loss_alpha'] * \
                tf.reduce_mean(input_tensor=tf.abs(batch_pos - x1))
            losses['ae_loss'] += FLAGS['l1_loss_alpha'] * \
                tf.reduce_mean(input_tensor=tf.abs(batch_pos - x2))

            batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(
                mask, [FLAGS['batch_size'] * 2, 1, 1, 1])], axis=3)

            # SNGAN
            pos_neg = D(batch_pos_neg, training=True)
            pos, neg = tf.split(pos_neg, 2)
            g_loss, d_loss = gan_hinge_loss(pos, neg)
            losses['g_loss'] = g_loss
            losses['d_loss'] = d_loss
            losses['g_loss'] = FLAGS['gan_loss_alpha'] * losses['g_loss']
            losses['g_loss'] += losses['ae_loss']
            # return losses
        gradients_of_generator = gen_tape.gradient(
            losses['g_loss'], G.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            losses['d_loss'], D.trainable_variables)
        g_optimizer.apply_gradients(
            zip(gradients_of_generator, G.trainable_variables))
        d_optimizer.apply_gradients(
            zip(gradients_of_discriminator, D.trainable_variables))

        # tensorboard
        if iter_idx > 0 and iter_idx % FLAGS['viz_max_out'] == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar('g_loss', losses['g_loss'], step=iter_idx)
                tf.summary.scalar('d_loss', losses['d_loss'], step=iter_idx)
                img = tf.reshape(batch_complete[0], (-1, 256, 256, 3))
                tf.summary.image("train", img, step=iter_idx)

    @tf.function
    def val_step(input_img, gt_img, mask_img):
        xin = (input_img)
        mask = (mask_img)
        xin = xin*(1.-mask)
        # mask already 0-1
        x1, x2, offset_flow = G(xin, mask, training=False)
        batch_predicted = x2

        losses = {}
        batch_complete = batch_predicted * mask + input_img * (1. - mask)
        return batch_complete
    epoch = 0
    epoch = tf.convert_to_tensor(epoch, dtype=tf.int64)
    for iter_idx in tqdm.tqdm(range(FLAGS['max_iters'])):
        iter_idx = tf.convert_to_tensor(iter_idx, dtype=tf.int64)
        train_step(iter_idx)
        if iter_idx > 0 and iter_idx % FLAGS['val_psteps'] == 0:
            tqdm.tqdm.write(str(iter_idx))
            psnr = []
            ds = val_ds.enumerate()
            idxx = 0
            for (input_img, gt_img, mask_img) in val_ds:
                batch_complete = (val_step(input_img, gt_img, mask_img))
                psnr.append(tf.image.psnr(
                    gt_img, batch_complete[0], max_val=1))
                with train_summary_writer.as_default():
                    img = batch_complete[:1]
                    tf.summary.image("val/%d" % idxx, img, step=epoch)
                idxx += 1
            psnr = tf.reduce_sum(psnr) / val_size
            with train_summary_writer.as_default():
                tf.summary.scalar('valPSNR', psnr, step=epoch)
            tqdm.tqdm.write(f'Cur {psnr} Best {best_psnr}')
            if (psnr > best_psnr):
                G.save_weights(os.path.join(model_dir, 'G'), save_format='tf')
                D.save_weights(os.path.join(model_dir, 'D'), save_format='tf')
                tqdm.tqdm.write('Saved models to %s' %
                                os.path.join(model_dir, 'G'))
                best_psnr = psnr
            epoch += 1
