import numpy
import tensorflow as tf

import network as net
import Fisher
import linalg

global_step = tf.Variable(0, trainable=False)
'''
learning_rate = tf.train.exponential_decay(10.0, global_step,
                                           100.0, 0.9)
'''
learning_rate = 1

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, 14, 14, 1])
Y = tf.placeholder(tf.float32, shape=[None, 10])


dd = 64
[X1], vs0 = net.conv_net([X], [1, 64, dd*2, dd], [3, 2, 2], [2, 2, 1], "D_conv", True)
X1 = tf.nn.relu(tf.reshape(X1, [-1, 4*4*dd]))
[X1] = net.normalize([X1])

[YY], vs1 = net.affine([X1], 4*4*dd, 10, "D_last")

theta = vs0 + vs1


def batch(batch_size):
    batch = mnist.train.next_batch(batch_size)
    x = numpy.reshape(batch[0], [batch_size, 28, 28, 1])
    x = x[:, 0:28:2, 0:28:2, :]
    y = batch[1]
    return x, y

costs = tf.nn.softmax_cross_entropy_with_logits(None, Y, YY)

'''

k = 10
X = tf.placeholder(tf.float32, shape=[None, k])
Y = tf.placeholder(tf.float32, shape=[None])

[YY], theta = net.affine_net([X], [k, k*2, k*2, 1], "D", False)


def batch(batch_size):
    x = numpy.random.normal(numpy.zeros([batch_size, k]), 1)
    y = numpy.mean(numpy.square(x), 1) > 1.0
    return x, y

costs = tf.nn.sigmoid_cross_entropy_with_logits(None, tf.reshape(Y, [-1, 1]), YY)
'''

cost = tf.reduce_mean(costs)

direction = tf.placeholder(tf.float32)
opt = tf.train.GradientDescentOptimizer(learning_rate)
step = opt.minimize(direction * cost, var_list=theta) #, global_step=global_step)


sess = tf.Session()


def converge(epochs = 1000, batch_size  = 64*2, direc = 1.0):
    sess.run(tf.assign(global_step, 0))
    for t in range(epochs):
        data_x, data_y = batch(batch_size)
        _, err = sess.run([step, cost], feed_dict={X:data_x, Y:data_y, direction:direc})
        print(err)
    return err

def scalar_prod(vs, ws):
    return sum([numpy.sum(v*w) for (v, w) in zip(vs, ws)])



sess.run(tf.global_variables_initializer())
'''
N = sum([numpy.prod(sh) for sh in sess.run([tf.shape(x) for x in theta])])
print(N)
'''
#theta2 = sess.run(theta)
print(converge())
theta0 = sess.run(theta)



sess.run(tf.global_variables_initializer())
print(converge())
theta1 = sess.run(theta)
'''
sess.run(tf.global_variables_initializer())
print(converge())
theta2 = sess.run(theta)

dxs = [v1 - v0 for (v0, v1) in zip(theta0, theta1)]

dys = [v2 - v0 for (v0, v2) in zip(theta0, theta2)]
nxs = scalar_prod(dxs, dxs)
sc = scalar_prod(dys, dxs) / nxs
print(sc)
dys = [dy - sc*dx for (dx, dy) in zip(dxs, dys)]
nys = scalar_prod(dys, dys)
sc = nxs / nys
dys = [dy*sc for dy in dys]
'''
#sess.run(reassign_ds)


N = 100
w = 30
surfaces = numpy.zeros([N, w, w])
data = batch(N)


#sess.run([add0, add1], feed_dict={d0:-0.5, d1:-0.5})

d = 2.0 / (w-1.0)

for ix in range(w):
    for iy in range(w):
        x = ix * 4.0 / (w - 1.0) - 1.5
        y = iy * 4.0 / (w - 1.0) - 1.5
        #sess.run([tf.assign(v, v0 + dx * x + dy * y) for (v, v0, dx, dy) in zip(theta, theta0, dxs, dys)])
        sess.run([tf.assign(v, v0 * x + v1 * y) for (v, v0, v1) in zip(theta, theta0, theta1)])
        #err = converge(100, 128, 1.0)

        err = sess.run(costs, feed_dict={X:data[0], Y:data[1]})
        surfaces[:, ix, iy] = numpy.reshape(err, [-1])

        print(ix, numpy.mean(err))
        
        #sess.run([add0], feed_dict={d0:d})
    #sess.run([add0, add1], feed_dict={d0:-d*w, d1:d})
    
sess.close()


import matplotlib.pyplot as P

num_p = 6; f, axes = P.subplots(2, num_p, sharex='col', sharey='row')
for i in range(num_p):
    surf = numpy.log(surfaces[i, :, :])
    axes[0, i].imshow(surf)
    axes[1, i].imshow(data[0][i, :, :, 0])
    #axes[1, i].contour(surf)

P.figure()

P.imshow(numpy.log(numpy.mean(surfaces, 0)))

P.show()

'''
P.hist([numpy.reshape(dx, [-1])for dx in dxs], histtype='barstacked',
        label=["W0", "b0", "W1", "b1", "W2", "b2", "W3", "b3"]);
P.legend();P.show()
'''

