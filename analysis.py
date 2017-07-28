import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import random
from scipy import misc
import time
import sys
#from drawCopy1 import viz_data, x, A, B, read_n, T
#from draw_eric import viz_data, x, A, B, read_n, T
from draw_eric_rewrite_filterbank import viz_data, x, A, B, read_n, T
#import load_input
#import load_trace

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)

saver = tf.train.Saver()

#data = load_trace.TraceData()
#data.get_test(1)

#data = load_input.InputData()
#data.get_test(1)

data = mnist.input_data.read_data_sets("mnist", one_hot=True).test

def random_image():
    """Get a random image from test set."""

    num_images = len(data.images)
    i = random.randrange(num_images)
    image_ar = np.array(data.images[i]).reshape(A, B)
    return image_ar#, data.labels[i]


def load_checkpoint(it):
    #path = "model_runs/blob_classification"
    #saver.restore(sess, "%s/drawmodel_%d.ckpt" % (path, it))
    #saver.restore(sess, "trace_draw/drawmodel.ckpt")
    saver.restore(sess, "model_runs/rewrite_filterbank/drawmodel.ckpt")


last_image = None


def read_img(it, new_image):
    batch_size = 1
    out = dict()
    global last_image
    if new_image or last_image is None:
        last_image = random_image()
    #img, label = last_image
    img = last_image
    flipped = np.flip(img.reshape(A, B), 0)
    out = {
        "img": flipped,
        #"label": label,
        "rects": list(),
        "rs": list(),
    }

    load_checkpoint(it)
    cs = sess.run(viz_data, feed_dict={x: img.reshape(batch_size, A*B)})

    for i in range(len(cs)):
        print('cs[i]["stats"]: ', cs[i]["stats"])
        #print(len(cs[i]["r"]))
        out["rs"].append(np.flip(cs[i]["r"].reshape(read_n, read_n), 0))
        out["rects"].append(stats_to_rect(cs[i]["stats"]))

    return out


def read_img2(it, new_image):
    """Read image with rewritten filterbanks."""

    batch_size = 1
    out = dict()
    global last_image
    if new_image or last_image is None:
        last_image = random_image()
    img = last_image
    flipped = np.flip(img.reshape(A, B), 0)
    out = {
        "img": flipped,
        "dots": list(),
    }

    load_checkpoint(it)
    cs = sess.run(viz_data, feed_dict={x: img.reshape(batch_size, A*B)})

    for i in range(len(cs)):
        mu_x = cs[i]["mu_x"]
        mu_y = cs[i]["mu_y"]
        delta = cs[i]["delta"]
        print("glimpse: ", i)

        print("mu_x: ")
        print(mu_x)
        print("mu_y: ")
        print(mu_y)

        print("delta: ")
        print(delta)

        out["dots"].append(list_to_dots(mu_x, mu_y))

    return out


def write_img(it, new_image):
    batch_size = 1
    out = dict()
    global last_image
    if new_image or last_image is None:
        last_image = random_image()
    #img, label = last_image
    img = last_image
    flipped = np.flip(img.reshape(A, B), 0)
    out = {
        #"label": label,
        "rects": list(),
        "c": list(),
    }

    load_checkpoint(it)
    cs = sess.run(viz_data, feed_dict={x: img.reshape(batch_size, A*B)})

    for i in range(len(cs)):
        out["c"].append(np.flip(cs[i]["c"].reshape(A, B), 0))
        out["rects"].append(stats_to_rect(cs[i]["w_stats"]))
        #print('cs[i]["stats"]: ')
        #print(cs[i]["stats"])
        #print('stats_to_rect[i]["stats"]: ')
        #print(stats_to_rect(cs[i]["stats"]))

    return out


def stats_to_rect(stats):
    """Draw attention window based on gx, gy, and delta."""

    gx, gy, delta = stats
    minY = A - gy + read_n/2.0 * delta
    maxY = B - gy - read_n/2.0 * delta

    minX = gx - read_n/2.0 * delta
    maxX = gx + read_n/2.0 * delta

    if minX < 1:
        minX = 1

    if maxY < 1:
        maxY = 1

    if maxX > A - 1:
        maxX = A - 1

    if minY > B - 1:
        minY = B - 1

    return dict(top=[int(minY)], bottom=[int(maxY)], left=[int(minX)], right=[int(maxX)])


def list_to_dots(mu_x, mu_y):
    """Draw filterbank based on mu_x and mu_y."""

    mu_x_list = mu_x * read_n
    mu_y_list = [val for val in mu_y for _ in range(0, read_n)]
 
    return dict(mu_x_list=mu_x_list, mu_y_list=mu_y_list)
