# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os
import time
import scipy.io
import tensorflow as tf
import numpy as np
import rawpy
import glob
import tqdm
import datetime
from PIL import Image

input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
checkpoint_dir = './checkpoint/Sony/'
result_dir = './result_Sony/'

os.makedirs(result_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

log_folder = 'logs/'
os.makedirs(log_folder, exist_ok=True)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = os.path.join(log_folder, current_time, 'train')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# get train IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

ps = 512  # patch size for training
save_freq = 500

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[0:5]


class upsample_and_concat(tf.keras.layers.Layer):
    def __init__(self, output_channels, in_channels):
        super(upsample_and_concat, self).__init__()
        pool_size = 2
        self.deconv = tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=pool_size, padding='SAME',
                                                      strides=pool_size, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02))
        self.output_channels = output_channels

    def call(self, x1, x2, training=True):
        deconv = self.deconv(x1)
        deconv_output = tf.concat([deconv, x2], 3)
        deconv_output.set_shape([None, None, None, self.output_channels * 2])
        return deconv_output


class NetWork(tf.keras.Model):
    def __init__(self):
        super(NetWork, self).__init__()
        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv1"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv2"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv3"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv4"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv5"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv6"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv7"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv8"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv9"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv10"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv6 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv11"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv12"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv7 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv11"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv12"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv8 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv13"),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv14"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv9 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv15"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv16"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(
                filters=12, kernel_size=1, strides=1, padding="SAME", name=self.name + "_conv17"),
        ])

        self.up6 = upsample_and_concat(256, 512)
        self.up7 = upsample_and_concat(128, 256)
        self.up8 = upsample_and_concat(64, 128)
        self.up9 = upsample_and_concat(32, 64)

    def call(self, x):
        conv1 = self.conv1(x)
        pool1 = tf.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME')
        conv2 = self.conv2(pool1)
        pool2 = tf.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME')
        conv3 = self.conv3(pool2)
        pool3 = tf.nn.max_pool2d(conv3, ksize=2, strides=2, padding='SAME')
        conv4 = self.conv4(pool3)
        pool4 = tf.nn.max_pool2d(conv4, ksize=2, strides=2, padding='SAME')
        conv5 = self.conv5(pool4)
        up6 = self.up6(conv5, conv4)
        conv6 = self.conv6(up6)
        up7 = self.up7(conv6, conv3)
        conv7 = self.conv7(up7)
        up8 = self.up8(conv7, conv2)
        conv8 = self.conv8(up8)
        up9 = self.up9(conv8, conv1)
        conv9 = self.conv9(up9)
        out = tf.nn.depth_to_space(input=conv9, block_size=2)
        return out


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


net = NetWork()
# net.load_weights(checkpoint_dir)

# Raw data takes long time to load. Keep them in memory after loaded.
gt_images = [None] * 6000
input_images = {}
input_images['300'] = [None] * len(train_ids)
input_images['250'] = [None] * len(train_ids)
input_images['100'] = [None] * len(train_ids)

g_loss = np.zeros((5000, 1))

allfolders = glob.glob(result_dir + '*0')
lastepoch = 0

learning_rate = 1e-4
lr = tf.Variable(name='lr', initial_value=learning_rate,
                 trainable=False, shape=[])
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
def var_list_fn(): return net.trainable_weights


for epoch in tqdm.tqdm(range(lastepoch, 4001)):
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    cnt = 0
    if epoch == 2000:
        lr = tf.Variable(name='lr', initial_value=1e-5,
                         trainable=False, shape=[])
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        tqdm.tqdm.write("Learning rate changed to 1e-5")
        # learning_rate = 1e-5

    for ind in tqdm.tqdm(np.random.permutation(len(train_ids))):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt += 1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(
                pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = input_images[str(
            ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_images[ind][:, yy * 2:yy *
                                  2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)

        with tf.GradientTape() as tape:
            output = net(input_patch)
            G_loss = tf.reduce_mean(input_tensor=tf.abs(output - gt_patch))
            gradients_of_generator = tape.gradient(
                G_loss, net.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients_of_generator, net.trainable_variables))
            output = np.minimum(np.maximum(output, 0), 1)
            g_loss[ind] = G_loss

        tqdm.tqdm.write("%d %d Avg Loss=%.3f Time=%.3f" % (
            epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))

    if epoch % save_freq == 0:

        temp = np.concatenate(
            (gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
        Image.fromarray((temp * 255).astype('uint8').clip(min=0, max=255)).save(
            result_dir + '%04d_%05d_00_train_%d.jpg' % (epoch, train_id, ratio))

        with train_summary_writer.as_default():
            img = tf.concat([output[:1], gt_patch[:1]], axis=2)
            tf.summary.image("train", img, step=(epoch))

    net.save_weights(checkpoint_dir, save_format='tf')
    tqdm.tqdm.write('Saved models to %s' % checkpoint_dir)
