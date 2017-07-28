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

        #print("mu_x: ")
        mu_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        mu_x = [1.0,
                7.7970576300000003,
                14.918401240000001,
                25.163368699999999,
                34.571436399999996,
                44.040457239999995,
                47.145239109999991,
                50.828340759999989,
                50.94945165,
                51.045047140000001,
                53.889735790000003,
                53.979128830000001,
                54.048518749999999,
                54.139084199999999,
                54.139084199999999,
                54.240633029999998,
                54.310951529999997,
                54.401024239999998,
                54.428910439999996,
                54.564423099999999,
                61.860441219999998,
                65.106687559999997,
                74.467402469999996,
                83.698637019999993,
                92.800889979999994]
        mu_x = [1.0,
                2.3594115260000001,
                3.7836802480000005,
                5.8326737400000006,
                7.7142872800000006,
                9.6080914480000015,
                10.229047822000002,
                10.965668152000003,
                10.989890330000001,
                11.009009428000002,
                11.577947158000002,
                11.595825766000003,
                11.609703750000003,
                11.627816840000003,
                11.627816840000003,
                11.648126606000003,
                11.662190306000001,
                11.680204848000001,
                11.685782088,
                11.712884620000001,
                13.172088244000001,
                13.821337512000001,
                15.693480494000001,
                17.539727404000001,
                19.360177996000001]
        #print(mu_x)
        #print("mu_y: ")
        mu_y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        mu_y = [1.0,
                7.7970576300000003,
                14.918401240000001,
                25.163368699999999,
                34.571436399999996,
                44.040457239999995,
                47.145239109999991,
                50.828340759999989,
                50.94945165,
                51.045047140000001,
                53.889735790000003,
                53.979128830000001,
                54.048518749999999,
                54.139084199999999,
                54.139084199999999,
                54.240633029999998,
                54.310951529999997,
                54.401024239999998,
                54.428910439999996,
                54.564423099999999,
                61.860441219999998,
                65.106687559999997,
                74.467402469999996,
                83.698637019999993,
                92.800889979999994]
        mu_y = [1.0,
                2.3594115260000001,
                3.7836802480000005,
                5.8326737400000006,
                7.7142872800000006,
                9.6080914480000015,
                10.229047822000002,
                10.965668152000003,
                10.989890330000001,
                11.009009428000002,
                11.577947158000002,
                11.595825766000003,
                11.609703750000003,
                11.627816840000003,
                11.627816840000003,
                11.648126606000003,
                11.662190306000001,
                11.680204848000001,
                11.685782088,
                11.712884620000001,
                13.172088244000001,
                13.821337512000001,
                15.693480494000001,
                17.539727404000001,
                19.360177996000001]
#print(mu_y)

        #print("delta: ")
        #print(delta)

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
