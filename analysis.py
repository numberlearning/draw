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
        mu_x = list(cs[i]["mu_x"])
        mu_y = list(cs[i]["mu_y"])
        delta = list(cs[i]["delta"])
        #print("glimpse: ", i)

        #print("mu_x: ")
        #mu_x = [  1.60994053,   4.36969948,   6.57750702,   8.34375381,   9.75675106,
        #        10.8871479,   11.79146671,  12.51492119,  13.09368515,  13.67244816,
        #        14.25121212,  14.82997513,  15.40873909,  15.98750305,  16.56626701,
        #        17.14502907,  17.72379303,  18.30255699,  19.02601242,  19.93033028,
        #        21.06072617,  22.47372437,  24.23997116,  26.4477787,   29.2075386 ]
        #print(mu_x)
        #print(np.array(mu_x).shape)
        #print("mu_y: ")
        #mu_y = [  1.60994053,   4.36969948,   6.57750702,   8.34375381,   9.75675106,
        #        10.8871479,   11.79146671,  12.51492119,  13.09368515,  13.67244816,
        #        14.25121212,  14.82997513,  15.40873909,  15.98750305,  16.56626701,
        #        17.14502907,  17.72379303,  18.30255699,  19.02601242,  19.93033028,
        #        21.06072617,  22.47372437,  24.23997116,  26.4477787,   29.2075386 ]
        #print(mu_y)
        #print(np.array(mu_y).shape)

        #print("delta: ")
        #delta = [2.7597599,   2.20780778,  1.76624632,  1.41299713,  1.13039756,  0.90431809,
        #        0.72345454,  0.5787636,   0.5787636,   0.5787636,   0.5787636,   0.5787636,   0.,
        #        0.5787636,   0.5787636,   0.5787636,   0.5787636,   0.5787636,   0.72345454,
        #        0.90431809,  1.13039756,  1.41299713,  1.76624632,  2.20780778,  2.7597599]
        #print(delta)
        #print(np.array(delta).shape)

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


def write_img2(it, new_image):
    """Write image with rewritten filterbanks."""

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
        "c": list(),
    }

    load_checkpoint(it)
    cs = sess.run(viz_data, feed_dict={x: img.reshape(batch_size, A*B)})

    for i in range(len(cs)):
        out["c"].append(np.flip(cs[i]["c"].reshape(A, B), 0))
        mu_x = list(cs[i]["w_mu_x"])
        mu_y = list(cs[i]["w_mu_y"])
        delta = list(cs[i]["w_delta"])
        out["dots"].append(list_to_dots(mu_x, mu_y))

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
