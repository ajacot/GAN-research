import numpy
import tensorflow as tf


def affine(xs, d0, d1, info):
    W = tf.Variable(tf.random_normal([d0, d1],
            stddev=1.0 / tf.sqrt(d0*1.0)),
                        name=info+"_W")
    b = tf.Variable(tf.zeros([d1]), name=info+"_b")
    return ([tf.matmul(x, W) + b for x in xs], [W, b])

def deconv(xs, width, stride, d0, d1, shape, info):
    W = tf.Variable(tf.random_normal([width, width, d1, d0],
            stddev=stride * 1.0 / (width*tf.sqrt(d0*1.0))),
                        name=info+"_W")
    b = tf.Variable(tf.zeros([d1]), name=info+"_b")
    return ([tf.nn.conv2d_transpose(x, W, shape + [d1], [1, stride, stride, 1], 'SAME') + b
                 for x in xs], [W, b])

def conv(xs, width, stride, d0, d1, info):
    W = tf.Variable(tf.random_normal([width, width, d0, d1],
            stddev=1.0 / (width*tf.sqrt(d0*1.0))),
                        name=info+"_W")
    b = tf.Variable(tf.zeros([d1]), name=info+"_b")
    return ([tf.nn.conv2d(x, W, [1, stride, stride, 1], 'SAME') + b
                 for x in xs], [W, b])


def deconv_net(xs, dims, widths, strides, batch_size, shape, info, apply_norm=True):
    variables = []
    for i in range(len(dims)-1):
        shape = [strides[i]*s for s in shape]
        xs, vs = deconv(xs, widths[i], strides[i], dims[i], dims[i+1],
                     [batch_size]+shape, info + str(i))
        variables = variables + vs
        if i < len(dims)-2:
            xs = [tf.nn.relu(x) for x in xs]
            if apply_norm:
                xs = normalize(xs, [0, 1, 2])
    return (xs, variables)


def conv_net(xs, dims, widths, strides, info, apply_norm=True):
    variables = []
    for i in range(len(dims)-1):
        xs, vs = conv(xs, widths[i], strides[i], dims[i], dims[i+1],
                     info + str(i))
        variables = variables + vs
        if i < len(dims)-2:
            xs = [tf.nn.relu(x) for x in xs]
            if apply_norm:
                xs = normalize(xs, [0, 1, 2])
    return (xs, variables)


def affine_net(xs, dims, info, apply_norm=True):
    variables = []
    for i in range(len(dims)-1):
        xs, vs = affine(xs, dims[i], dims[i+1], info + str(i))
        variables = variables + vs
        if i < len(dims)-2:
            xs = [tf.nn.relu(x) for x in xs]
            if apply_norm:
                xs = normalize(xs, [0])
    return (xs, variables)


'''
def normalize(X, dims=[0], variance_epsilon=0.01, block=False):
    mean, var = tf.nn.moments(X, dims, keep_dims=True)
    if block:
        mean = tf.stop_gradient(mean)
        var = tf.stop_gradient(var)
    return (X - mean) / tf.sqrt(var + variance_epsilon)
'''
def normalize(X, dims=[0], variance_epsilon=0.01):
    mean, var = tf.nn.moments(X[0], dims, keep_dims=True)
    return [tf.nn.batch_normalization(x, mean, var, None, None, variance_epsilon) for x in X]

def get_subset(net, skip = 10):
    return tf.concat([
            tf.reshape(param, [-1])[0:-1:skip]
            for param in net], 0)
