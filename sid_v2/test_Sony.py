# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io
import tensorflow as tf
import numpy as np
import rawpy
import glob
import tqdm
from PIL import Image
# from train_Sony import pack_raw, upsample_and_concat, NetWork

input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
checkpoint_dir = './checkpoint/Sony/'
result_dir = './result_Sony/'

# get test IDs
test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]


# sess = tf.compat.v1.Session()
# in_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 4])
# gt_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3])
# out_image = network(in_image)

# saver = tf.compat.v1.train.Saver()
# sess.run(tf.compat.v1.global_variables_initializer())
# ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
# if ckpt:
#     print('loaded ' + ckpt.model_checkpoint_path)
#     saver.restore(sess, ckpt.model_checkpoint_path)

def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.random.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(input=x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

class NetWork(tf.keras.Model):
    def __init__(self):
        super(NetWork, self).__init__()
        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv1"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv2"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv3"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv4"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv5"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv6"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv7"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv8"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv9"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv10"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv6 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv11"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv12"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv7 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv11"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv12"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv8 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv13"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv14"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv9 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv15"),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="SAME", name=self.name + "_conv16"),
            tf.keras.layers.ReLU(negative_slope=0.2),
        ])

        self.conv10 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=12, kernel_size=1, strides=1, padding="SAME", name=self.name + "_conv16"),
        ])

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
        # pool5 = tf.nn.max_pool2d(conv5, ksize=2, strides=2, padding='SAME')
        up6 = upsample_and_concat(conv5, conv4, 256, 512)
        conv6 = self.conv6(up6)
        up7 = upsample_and_concat(conv6, conv3, 128, 256)
        conv7 = self.conv7(up7)
        up8 = upsample_and_concat(conv7, conv2, 64, 128)
        conv8 = self.conv8(up8)
        up9 = upsample_and_concat(conv8, conv1, 32, 64)
        conv9 = self.conv9(up9)
        conv10 = self.conv10(conv9)
        out = tf.nn.depth_to_space(input=conv10, block_size=2)
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

net.load_weights(checkpoint_dir)

# if not os.path.isdir(result_dir + 'final/'):
os.makedirs(result_dir + 'final/', exist_ok=True)
psnr = 0
cnt = 0

for test_id in tqdm.tqdm(test_ids):
    # test the first image in each sequence
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
    for k in tqdm.tqdm(range(len(in_files))):
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        input_full = np.minimum(input_full, 1.0)

        output = net(input_full)
        # output = sess.run(out_image, feed_dict={in_image: input_full})
        output = np.minimum(np.maximum(output, 0), 1)

        cur_psnr = tf.image.psnr(gt_full, output, max_val=1)
        psnr += cur_psnr
        tqdm.tqdm.write(f"{in_fn} PSNR: {cur_psnr}")
        cnt += 1

        output = output[0, :, :, :]
        gt_full = gt_full[0, :, :, :]
        scale_full = scale_full[0, :, :, :]
        scale_full = scale_full * np.mean(gt_full) / np.mean(
            scale_full)  # scale the low-light image to the same mean of the groundtruth

        Image.fromarray((output * 255).astype('uint8').clip(min=0, max=255)).save(
        # scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/%5d_00_%d_out.png' % (test_id, ratio))
        Image.fromarray((scale_full * 255).astype('uint8').clip(min=0, max=255)).save(
        # scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/%5d_00_%d_scale.png' % (test_id, ratio))
        Image.fromarray((gt_full * 255).astype('uint8').clip(min=0, max=255)).save(
        # scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/%5d_00_%d_gt.png' % (test_id, ratio))

print(psnr * 1.0 / cnt)
