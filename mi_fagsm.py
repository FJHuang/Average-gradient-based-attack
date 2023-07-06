import tensorflow as tf 
import numpy as np 
import scipy
import os
import glob
import csv

from nets import inception, resnet_v2
from PIL import Image
from scipy.misc import imread, imsave, imresize
import tensorflow.contrib.slim as slim
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
slim = tf.contrib.slim

tf.flags.DEFINE_string('checkpoint_path', " ", 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string('input_dir', " ", 'Input directory with images.')
tf.flags.DEFINE_string('output_dir'," ", 'Output directory with images.')
tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')
tf.flags.DEFINE_integer('batch_size', 5, 'How many images process at one time.')
tf.flags.DEFINE_integer('num_classes', 1001, 'Number of Classes.')
tf.flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')
tf.flags.DEFINE_integer('momentum', 1, 'momentum.')
FLAGS = tf.flags.FLAGS
tf.app.flags.DEFINE_string('f', '', 'kernel')

def _check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_images(dev_dir, input_dir, batch_shape):
    images = np.zeros(batch_shape)
    labels = np.zeros(batch_shape[0], dtype=np.int32)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    with open(dev_dir, 'r+',encoding='gbk') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = os.path.join(input_dir, row['Filename'])
            with tf.gfile.Open(filepath, "rb") as f:
                r_img = imread(f, mode='RGB')
                image = imresize(r_img, [299, 299]).astype(np.float) / 255.0
            images[idx, :, :, :] = image * 2.0 -1.0
            labels[idx] = int(row['Label'])
            filenames.append(os.path.basename(filepath))
            idx += 1
            if idx == batch_size:
                yield filenames, images, labels + 1
                filenames = []
                images = np.zeros(batch_shape)
                labels = np.zeros(batch_shape[0], dtype=np.int32)
                idx = 0
        if idx > 0:
            yield filenames, images, labels + 1

def grad_X(x,y):
    tf.get_variable_scope().reuse_variables()
    with slim.arg_scope(inception.inception_v4_arg_scope()):
        logits, end_points = inception.inception_v4(x, num_classes=FLAGS.num_classes, is_training=False)
    #logits = (end_points['Logits'])
    cross_entropy = tf.losses.softmax_cross_entropy(y,logits,label_smoothing=0.0,weights=1.0)
    noise = tf.gradients(cross_entropy, x)[0]
    return noise
    
def graph_arrack(x, y, x_max, x_min, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    momentum = FLAGS.momentum
    num_iter = FLAGS.num_iter
    alpha = eps / FLAGS.num_iter
    x_advs = list()
    g_i = list()
    x_adv = x
    noise = grad
    tf.get_variable_scope().reuse_variables()
    for t in range(0,num_iter):
        x_advs.append(x_adv)
        gs = tf.zeros(shape=[FLAGS.batch_size, 299, 299, 3])
        for i in range(0,t):
            x_adv_i = x_advs[t-1] + alpha * g_i[i]
            gs = gs + grad_X(x_adv_i,y)
        g_ = (gs + grad_X(x_adv,y))/(t+1)
        g_ = g_ / tf.reduce_mean(tf.abs(g_), [1, 2, 3], keep_dims=True)
        g_i.append(g_)
        noise = momentum * noise + g_
        x_adv = x_adv + alpha * tf.sign(noise)
        x_adv = tf.clip_by_value(x_adv, x_min, x_max)       
    return x_adv    
    

def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        with open(os.path.join(output_dir, filename), 'wb+') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            r_img = imresize(img, [299, 299])
            Image.fromarray(r_img).save(f, format='PNG')

def main(input_dir, output_dir):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, 299, 299, 3]
    _check_or_create_dir(output_dir)
    dev_dir = " "
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)
        
        with slim.arg_scope(inception.inception_v4_arg_scope()):
            logits, end_points = inception.inception_v4(x_input, num_classes=1001, is_training=False)
        score = tf.nn.softmax(logits,name='pre')
        pred_labels = tf.argmax(score, 1)
        y = tf.one_hot(pred_labels, FLAGS.num_classes)
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        x_adv = graph_arrack(x_input, y, x_max, x_min, grad)
        # Run computation
        saver = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        with tf.Session() as sess: 
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, FLAGS.checkpoint_path)
            for filenames, raw_images, true_labels in load_images(dev_dir, input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: raw_images})
                save_images(adv_images, filenames, output_dir)

if __name__=='__main__':
    main(FLAGS.input_dir, FLAGS.output_dir)
