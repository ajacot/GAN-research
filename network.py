import numpy
import tensorflow as tf


def affine(xs, d0, d1, info):
    W = tf.Variable(tf.random_normal([d0, d1],
                        stddev=1.0 / tf.sqrt(d0*1.0)),
                        name=info+"_W")
    #b = tf.Variable(tf.zeros([d1]), name=info+"_b")
    b = tf.Variable(tf.random_normal([d1]), name=info+"_b")
    #return ([(tf.matmul(x, W)+b) / tf.sqrt(d0*1.0) for x in xs], [W, b])
    return ([(tf.matmul(x, W)+b) for x in xs], [W, b])

def deconv(xs, width, stride, d0, d1, shape, info):
    W = tf.Variable(tf.random_normal([width, width, d1, d0]),
                        name=info+"_W")
    #stddev=stride * 1.0 / (width*tf.sqrt(d0*1.0))),
    b = tf.Variable(tf.zeros([d1]), name=info+"_b")
    return ([(tf.nn.conv2d_transpose(x, W, shape + [d1], [1, stride, stride, 1], 'SAME')+b) / (width*tf.sqrt(d0*1.0))
                 for x in xs], [W, b])

def conv(xs, width, stride, d0, d1, info):
    W = tf.Variable(tf.random_normal([width, width, d0, d1]),
                        name=info+"_W")
    # stddev=1.0 / (width*tf.sqrt(d0*1.0))),
    b = tf.Variable(tf.zeros([d1]), name=info+"_b")
    return ([(tf.nn.conv2d(x, W, [1, stride, stride, 1], 'SAME') + b) / (width*tf.sqrt(d0*1.0))
                 for x in xs], [W, b])


def deconv_net(xs, dims, widths, strides, batch_size, shape,
               info, apply_norm=True, non_lin=tf.nn.relu):
    variables = []
    for i in range(len(dims)-1):
        shape = [strides[i]*s for s in shape]
        xs, vs = deconv(xs, widths[i], strides[i], dims[i], dims[i+1],
                     [batch_size]+shape, info + str(i))
        variables = variables + vs
        if i < len(dims)-2:
            xs = [non_lin(x) for x in xs]
            if apply_norm:
                xs = normalize(xs, [0, 1, 2])
    return (xs, variables)


def conv_net(xs, dims, widths, strides,
             info, apply_norm=True, non_lin=tf.nn.relu):
    variables = []
    for i in range(len(dims)-1):
        xs, vs = conv(xs, widths[i], strides[i], dims[i], dims[i+1],
                     info + str(i))
        variables = variables + vs
        if i < len(dims)-2:
            xs = [non_lin(x) for x in xs]
            if apply_norm:
                xs = normalize(xs, [0, 1, 2])
    return (xs, variables)


def affine_net(xs, dims,
               info, apply_norm=True, non_lin=tf.nn.relu):
    variables = []
    for i in range(len(dims)-1):
        xs, vs = affine(xs, dims[i], dims[i+1], info + str(i))
        variables = variables + vs
        if i < len(dims)-2:
            xs = [non_lin(x) for x in xs]
            if apply_norm:
                xs = normalize(xs, [0])
                #xs, step = normalize2(xs, [0], 1, dims[i+1])
                #step_norm = step_norm + [step]
    return (xs, variables)


def normalize(X, dims=[0], variance_epsilon=0.01):
    mean, var = tf.nn.moments(X[0], dims, keep_dims=True)
    mean = tf.stop_gradient(mean)
    var = tf.stop_gradient(var)
    return [tf.nn.batch_normalization(x, mean, var, None, None, variance_epsilon) for x in X]

def normalize2(X, dims=[0], n=2, d=0, variance_epsilon=0.01):
    mean, var = tf.nn.moments(X[0], dims, keep_dims=True)
    mean = tf.stop_gradient(mean)
    var = tf.stop_gradient(var)
    X = [tf.nn.batch_normalization(x, mean, var, None, None, variance_epsilon) for x in X]

    #dirs = tf.Variable(X[0][:, 0:n])
    dirs = tf.Variable(tf.random_normal([d, n], 0.0, 1.0 / tf.sqrt(tf.cast(d, tf.float32))))
    ddirs = tf.qr(dirs)[0]
    affinities = tf.matmul(X[0], ddirs)
    new_dirs = tf.matmul(tf.transpose(X[0]), affinities)

    sc = (1.0 / (tf.norm(affinities, axis=0) + variance_epsilon) - 1.0)
    sc = tf.stop_gradient(sc)
    M = tf.matmul(ddirs * sc, ddirs, transpose_b=True)
    
    X = [x + tf.matmul(x, M) for x in X]

    step = dirs.assign(tf.qr(new_dirs)[0])
    
    return X, step


def get_subset(net, skip = 10):
    return tf.concat([
            tf.reshape(param, [-1])[0:-1:skip]
            for param in net], 0)


right_90 = numpy.array([[0.0, -1.0], [1.0, 0.0]], numpy.float32)

def sparse_deconv2D(vs, ps, rs, ints, n, k, d0, d1):
    # features vs :: batch_size, n, d0
    # positions ps :: batch_size, n, d
    # rot_scalings rs :: batch_size, n, d
    # intensities ints :: batch_size, n, 1
    d = 2
    
    def apply_rs(x):
        return (tf.reshape(rs[:, :, 0], [-1, n, 1, 1]) * x
            + tf.reshape(rs[:, :, 1], [-1, n, 1, 1]) * tf.tensordot(x, right_90, [[3], [0]]))

    theta = []
    dims = [d0, d0 * 3, d1*k + d*k + d*k + k]
    for i in range(len(dims)-1):
        #vs = tf.nn.relu(vs)
        #vs, = normalize([vs], [0, 1])
        W = tf.Variable(tf.random_normal([dims[i], dims[i+1]]),name="W")
        b = tf.Variable(tf.zeros([dims[i+1]]), name="b")
        vs = (tf.tensordot(vs, W, [[2], [0]])+b) / tf.sqrt(dims[i]*1.0)
        theta = theta + [W, b]

    new_vs, new_ps, new_rs, new_ints = tf.split(vs, [d1*k, d*k, d*k, k], 2)
    new_ps = tf.reshape(new_ps, [-1, n, k, 2])
    new_rs = tf.reshape(new_rs, [-1, n, k, 2]) * 0.3
    
    new_vs = tf.reshape(new_vs, [-1, n*k, d1])
    
    new_ps = tf.reshape(apply_rs(new_ps) + tf.reshape(ps, [-1, n, 1, d]), [-1, n*k, d])
    new_rs = tf.reshape(apply_rs(new_rs), [-1, n*k, d])
    new_ints = tf.reshape(new_ints * ints, [-1, n*k, 1])
    
    return new_vs, new_ps, new_rs, new_ints, theta

def sparse_deconv_net(vs, ps, rs, ints, n0, ks, dims):
    n = n0
    theta = []
    for i in range(len(ks)):
        vs, ps, rs, ints, thet = sparse_deconv2D(vs, ps, rs, ints, n, ks[i], dims[i], dims[i+1])
        theta = theta + thet
        n = n*ks[i]

    return vs, ps, rs, ints, theta

def sparse_to_image(vs, ps, rs, ints, w):
    xs = tf.linspace(-1.0, 1.0, w)
    ys = tf.linspace(-1.0, 1.0, w)

    norms = tf.norm(rs, axis=2)
    mask_x = tf.exp(-1500.0*norms*tf.square(ps[:, :, 0] - tf.reshape(xs, [-1, 1, 1]))) * (0.5 + 0.5*tf.tanh(ints[:, :, 0]))
    mask_y = tf.exp(-1500.0*norms*tf.square(ps[:, :, 1] - tf.reshape(ys, [-1, 1, 1])))

    return tf.tensordot(mask_y, mask_x, [[1, 2], [1, 2]])
'''
d0 = 10

vs = tf.placeholder(tf.float32, [1, 1, d0])
ps = tf.placeholder(tf.float32, [1, 1, 2])
rs = tf.placeholder(tf.float32, [1, 1, 2])
ints = tf.placeholder(tf.float32, [1, 1, 1])

new_vs, new_ps, new_rs, new_ints, theta = sparse_deconv_net(vs, ps, rs, ints, 1, [5, 5, 5], [d0, d0, d0, 2])

image = sparse_to_image(new_vs, new_ps, new_rs, new_ints, 28)

import matplotlib.pyplot as P

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    v0 = numpy.random.normal(numpy.zeros([1, 1, d0]))
    p0 = numpy.zeros([1, 1, 2])
    r0 = numpy.random.normal(numpy.zeros([1, 1, 2])) * 0.5
    int0 = numpy.ones([1, 1, 1])

    v1, p1, r1, int1, img = sess.run([new_vs, new_ps, new_rs, new_ints, image],
                                feed_dict={vs:v0, ps:p0, rs:r0, ints:int0})
    P.imshow(img, extent=(-1.0, 1.0, 1.0, -1.0))
    P.quiver(p1[0, :, 0], p1[0, :, 1], r1[0, :, 0], r1[0, :, 1])

    P.show()    
'''
