import numpy
import tensorflow as tf

import network as net
import Fisher
import linalg

global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(8.0, global_step,
                                           1000.0, 0.9999)

'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, 14, 14, 1])
Y = tf.placeholder(tf.float32, shape=[None])


[X1], vs0 = net.conv_net([X], [1, 8, 16], [3, 2], [2, 2], "D_conv", True)
X1 = tf.nn.relu(tf.reshape(X1, [-1, 4*4*16]))

[YY], vs1 = net.affine([X1], 4*4*16, 1, "D_last")

theta = vs0 + vs1


def batch(batch_size):
    batch = mnist.train.next_batch(batch_size)
    x = numpy.reshape(batch[0], [batch_size, 28, 28, 1])
    x = x[:, 0:28:2, 0:28:2, :]
    y = batch[1] == 0
    return x, y
'''

k = 10
X = tf.placeholder(tf.float32, shape=[None, k])
Y = tf.placeholder(tf.float32, shape=[None])

[YY], theta = net.affine_net([X], [k, k*2, k*2, 1], "D", False)

d0 = tf.placeholder(tf.float32)
d1 = tf.placeholder(tf.float32)

i0 = 0
j0 = 1

i1 = 3
j1 = 4

d0W0 = tf.Variable(theta[0][:, j0] - theta[0][:, i0])
d0b0 = tf.Variable(theta[1][j0] - theta[1][i0])
d0W1 = tf.Variable(theta[2][j0, :] - theta[2][i0, :])

d1W0 = tf.Variable(theta[0][:, j1] - theta[0][:, i1])
d1b0 = tf.Variable(theta[1][j1] - theta[1][i1])
d1W1 = tf.Variable(theta[2][j1, :] - theta[2][i1, :])

reassign_ds = [v.initializer for v in [d0W0, d0b0, d0W1, d1W0, d1b0, d1W1]]

add0 = [theta[0][:, i0].assign(theta[0][:, i0] + d0*d0W0),
        theta[0][:, j0].assign(theta[0][:, j0] - d0*d0W0),
        theta[1][i0].assign(theta[1][i0] + d0*d0b0),
        theta[1][j0].assign(theta[1][j0] - d0*d0b0),
        theta[2][i0, :].assign(theta[2][i0, :] + d0*d0W1),
        theta[2][j0, :].assign(theta[2][j0, :] - d0*d0W1)]

add1 = [theta[0][:, i1].assign(theta[0][:, i1] + d1*d1W0),
        theta[0][:, j1].assign(theta[0][:, j1] - d1*d1W0),
        theta[1][i1].assign(theta[1][i1] + d1*d1b0),
        theta[1][j1].assign(theta[1][j1] - d1*d1b0),
        theta[2][i1, :].assign(theta[2][i1, :] + d1*d1W1),
        theta[2][j1, :].assign(theta[2][j1, :] - d1*d1W1)]

def batch(batch_size):
    x = numpy.random.normal(numpy.zeros([batch_size, k]), 1)
    y = numpy.mean(numpy.square(x), 1) > 0.5
    return x, y

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.reshape(Y, [-1, 1]), YY))

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(None, Y, YY))

direction = tf.placeholder(tf.float32)
opt = tf.train.GradientDescentOptimizer(learning_rate)
step = opt.minimize(direction * cost, var_list=theta) #, global_step=global_step)


sess = tf.Session()

N = sum([numpy.prod(sh) for sh in sess.run([tf.shape(x) for x in theta])])
print(N)

data_x, data_y = batch(20)
data_x2, data_y2 = batch(20)

def converge(epochs = 600, batch_size  = 64, direc = 1.0):
    sess.run(tf.assign(global_step, 0))
    for t in range(epochs):
        data_x, data_y = batch(batch_size)
        _, err = sess.run([step, cost], feed_dict={X:data_x, Y:data_y, direction:direc})
        print(err)
    return err

def scalar_prod(vs, ws):
    return sum([numpy.sum(v*w) for (v, w) in zip(vs, ws)])


def swap_neurons(W0, b0, W1, i, j, d):
    W0_mod = W0


sess.run(tf.global_variables_initializer())
#theta2 = sess.run(theta)
print(converge())
theta0 = sess.run(theta)



sess.run(tf.global_variables_initializer())
print(converge())
theta1 = sess.run(theta)

sess.run(tf.global_variables_initializer())
print(converge())
theta2 = sess.run(theta)

dxs = [v1 - v0 for (v0, v1) in zip(theta0, theta1)]

dys = [v2 - v0 for (v0, v2) in zip(theta0, theta2)]
sc = scalar_prod(dys, dxs) / scalar_prod(dxs, dxs)
print(sc)
dys = [dy - sc*dx for (dx, dy) in zip(dxs, dys)]

#sess.run(reassign_ds)


'''
sess.run(tf.global_variables_initializer())
theta0 = sess.run(theta)
sess.run(tf.global_variables_initializer())
dxs = sess.run(theta)
sess.run(tf.global_variables_initializer())
dys = sess.run(theta)
'''


w = 30
surface = numpy.zeros([w, w])
surface2 = numpy.zeros([w, w])

sess.run([add0, add1], feed_dict={d0:-0.5, d1:-0.5})

d = 2.0 / (w-1.0)

for ix in range(w):
    for iy in range(w):
        x = ix * 2.0 / (w - 1.0) - 0.5
        y = iy * 2.0 / (w - 1.0) - 0.5
        sess.run([tf.assign(v, v0 + dx * x + dy * y) for (v, v0, dx, dy) in zip(theta, theta0, dxs, dys)])
        #err = converge(100, 128, 1.0)
        err = sess.run(cost, feed_dict={X:data_x, Y:data_y})
        err2 = sess.run(cost, feed_dict={X:data_x2, Y:data_y2})
        surface[ix, iy] = err
        surface2[ix, iy] = err2
        #cut[ix] = err
        print(ix, err)
        
        #sess.run([add0], feed_dict={d0:d})
    #sess.run([add0, add1], feed_dict={d0:-d*w, d1:d})
    
sess.close()


import matplotlib.pyplot as P

f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = P.subplots(3, 2, sharex='col', sharey='row')
ax1.imshow(numpy.log(surface))
ax2.contour(numpy.log(surface))
ax3.imshow(numpy.log(surface2))
ax4.contour(numpy.log(surface2))
ax5.imshow(numpy.log((surface + surface2)/2.0))
ax6.contour(numpy.log((surface + surface2)/2.0))

#P.imshow(numpy.log(surface))
P.show()



