import numpy
import tensorflow as tf


def mnist(one_hot=True):
    X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    if one_hot:
        Y = tf.placeholder(tf.float32, shape=[None, 10])
    else:
        Y = tf.placeholder(tf.float32, shape=[None])

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=one_hot)

    def batch(batch_size):
        x, y = mnist.train.next_batch(batch_size)
        x = numpy.reshape(x, [batch_size, 28, 28, 1])
        return {X:x, Y:y}

    return X, Y, batch


def half_mnist(one_hot=True):
    X = tf.placeholder(tf.float32, shape=[None, 14, 14, 1])
    if one_hot:
        Y = tf.placeholder(tf.float32, shape=[None, 10])
    else:
        Y = tf.placeholder(tf.float32, shape=[None])

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=one_hot)

    def batch(batch_size):
        x, y = mnist.train.next_batch(batch_size)
        x = numpy.reshape(x, [batch_size, 28, 28, 1])[:, 0:28:2, 0:28:2, :]
        return {X:x, Y:y}

    return X, Y, batch

def celeb_A():
    '''
    def parse_line(line):
        elems = tf.decode_csv(line, [[""]] + [[0] for _ in range(40)], ' ')
        filename = elems[0]
        features = elems[1:]
        img = tf.image.decode_jpeg(tf.read_file(
            tf.string_join(["CelebA/img_align_celeba/img_align_celeba/", filename])), channels=3)
        return img, features
        
    dataset = (tf.data.TextLineDataset("CelebA/Anno/list_attr_celeba.txt") # Read text file
        .skip(2)
        .map(parse_line)
        .shuffle(buffer_size=64))'''

    mid_h = 109
    mid_w = 89
    x0 = mid_w - 32
    x1 = mid_w + 32
    y0 = mid_h - 4
    y1 = mid_h + 60
    
    def parse_line(line):
        elems = tf.decode_csv(line, [[""], [0]], ' ')
        filename = elems[0]
        valid = elems[1]
        img = tf.image.decode_jpeg(tf.read_file(
            tf.string_join(["CelebA/img_align_celeba/img_align_celeba/", filename])), channels=3)
        return tf.cast(img[y0:y1, x0:x1, :], tf.float32) / 256.0 # , valid
        
    dataset = (tf.data.TextLineDataset("CelebA/Eval/list_eval_partition.txt") # Read text file
        .skip(2)
        .map(parse_line)
        .shuffle(buffer_size=64))

    return dataset


def random_normal(d):
    Z = tf.placeholder(tf.float32, shape=[None, d])
    def batch(batch_size):
        z = numpy.random.normal(numpy.zeros([batch_size, d]))
        return {Z:z}
    return Z, batch
