import os
import glob
import yaml

import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintGenerator, InpaintDiscriminator, gan_hinge_loss

def prepare_for_training(ds, shuffle_buffer_size=1000, batch_size=1, repeat=False):
    # https://www.tensorflow.org/tutorials/load_data/images
    # places2 toooooo big cant cache
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    if repeat:
        # Repeat forever for train
        ds = ds.repeat()
    ds = ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_image(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # dont resize here, leave resize in the train loop to generate patches on-the-fly
    return img

def process_path(file_path, resize=False):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    if resize:
        img = tf.image.resiz(img, [256, 256])
    return img

def get_train_iter():
    data_dir = 'examples/places2/'
    list_ds = tf.data.Dataset.list_files(str(data_dir + '*_output.png'))
    new_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = prepare_for_training(new_ds, repeat=True)
    return iter(train_ds)

def get_val_iter():
    data_dir = 'examples/places2/'
    list_ds = tf.data.Dataset.list_files(str(data_dir + '*_output.png'))
    new_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = prepare_for_training(new_ds, repeat=False)
    return iter(train_ds)

if __name__ == "__main__":
    # training data
    # FLAGS = ng.Config('inpaint.yml')
    FLAGS = yaml.load(open('inpaint.yml', 'r'), Loader=yaml.FullLoader)
    img_shapes = FLAGS.img_shapes # 256x256
    with open(FLAGS.data_flist[FLAGS.dataset][0]) as f:
        fnames = f.read().splitlines()
    
    # data = ng.data.DataFromFNames(
    #     fnames, img_shapes, random_crop=FLAGS.random_crop,
    #     nthreads=FLAGS.num_cpus_per_job)
    # images = data.data_pipeline(FLAGS.batch_size)
    # main model
    # model = InpaintCAModel()
    # g_vars, d_vars, losses = model.build_graph_with_losses(FLAGS, images)
    # validation images
    # if FLAGS.val:
    #     with open(FLAGS.data_flist[FLAGS.dataset][1]) as f:
    #         val_fnames = f.read().splitlines()
    #     # progress monitor by visualizing static images
    #     for i in range(FLAGS.static_view_size):
    #         static_fnames = val_fnames[i:i+1]
    #         static_images = ng.data.DataFromFNames(
    #             static_fnames, img_shapes, nthreads=1,
    #             random_crop=FLAGS.random_crop).data_pipeline(1)
    #         static_inpainted_images = model.build_static_infer_graph(
    #             FLAGS, static_images, name='static_view/%d' % i)
    # training settings

    G = InpaintGenerator()
    D = InpaintDiscriminator()
    lr = tf.Variable(name='lr', initial_value=1e-4, trainable=False, shape=[])
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta1=0.5, beta2=0.999)
    g_optimizer = d_optimizer

    # data
    train_iter = get_train_iter()
    val_iter = get_val_iter()
    # TODO: val images should be static and generated before running

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # disable FLAGS.guided
            batch_data = next(train_iter)
            batch_pos = batch_data / 127.5 - 1.
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
            batch_incomplete = batch_pos*(1.-mask)
            xin = batch_incomplete
            x1, x2, offset_flow = G(xin, mask, training=True)
            batch_predicted = x2

            losses = {}
            batch_complete = batch_predicted * mask + batch_incomplete * (1. - mask)
            losses['ae_loss'] = FLAGS.l1_loss_alpha * tf.reduce_mean(input_tensor=tf.abs(batch_pos - x1))
            losses['ae_loss'] += FLAGS.l1_loss_alpha * tf.reduce_mean(input_tensor=tf.abs(batch_pos - x2))

            batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [FLAGS.batch_size * 2, 1, 1, 1])], axis=3)

            # SNGAN
            pos_neg = D(batch_pos_neg, training=training)
            pos, neg = tf.split(pos_neg, 2)
            g_loss, d_loss = gan_hinge_loss(pos, neg)
            losses['g_loss'] = g_loss
            losses['d_loss'] = d_loss
            losses['g_loss'] = FLAGS.gan_loss_alpha * losses['g_loss']
            losses['g_loss'] += losses['ae_loss']
            # return losses
        gradients_of_generator = gen_tape.gradient(losses['g_loss'], G.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(losses['d_loss'], D.trainable_variables)
        g_optimizer.apply_gradients(zip(gradients_of_generator, G.trainable_variables))
        d_optimizer.apply_gradients(zip(gradients_of_discriminator, D.trainable_variables))


    # train discriminator with secondary trainer, should initialize before
    # primary trainer.
    # discriminator_training_callback = ng.callbacks.SecondaryTrainer(
    # discriminator_training_callback = ng.callbacks.SecondaryMultiGPUTrainer(
    #     num_gpus=FLAGS.num_gpus_per_job,
    #     pstep=1,
    #     optimizer=d_optimizer,
    #     var_list=d_vars,
    #     max_iters=1,
    #     grads_summary=False,
    #     graph_def=multigpu_graph_def,
    #     graph_def_kwargs={
    #         'model': model, 'FLAGS': FLAGS, 'data': data, 'loss_type': 'd'},
    # )
    # # train generator with primary trainer
    # # trainer = ng.train.Trainer(
    # trainer = ng.train.MultiGPUTrainer(
    #     num_gpus=FLAGS.num_gpus_per_job,
    #     optimizer=g_optimizer,
    #     var_list=g_vars,
    #     max_iters=FLAGS.max_iters,
    #     graph_def=multigpu_graph_def,
    #     grads_summary=False,
    #     gradient_processor=None,
    #     graph_def_kwargs={
    #         'model': model, 'FLAGS': FLAGS, 'data': data, 'loss_type': 'g'},
    #     spe=FLAGS.train_spe,
    #     log_dir=FLAGS.log_dir,
    # )
    # # add all callbacks
    # trainer.add_callbacks([
    #     discriminator_training_callback,
    #     ng.callbacks.WeightsViewer(),
    #     ng.callbacks.ModelRestorer(trainer.context['saver'], dump_prefix=FLAGS.model_restore+'/snap', optimistic=True),
    #     ng.callbacks.ModelSaver(FLAGS.train_spe, trainer.context['saver'], FLAGS.log_dir+'/snap'),
    #     ng.callbacks.SummaryWriter((FLAGS.val_psteps//1), trainer.context['summary_writer'], tf.compat.v1.summary.merge_all()),
    # ])
    # # launch training
    # trainer.train()
