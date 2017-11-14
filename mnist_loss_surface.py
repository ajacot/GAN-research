import numpy
import tensorflow as tf

import network as net
import Fisher

global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(0.1, global_step,
                                           50.0, 0.5)

'''
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

def batch(batch_size):
    x = numpy.random.normal(numpy.zeros([batch_size, k]), 1)
    y = numpy.mean(numpy.square(x), 1) > 1.0 / k
    return x, y


cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.reshape(Y, [-1, 1]), YY))

direction = tf.placeholder(tf.float32)
opt = tf.train.GradientDescentOptimizer(learning_rate)
norm_theta = Fisher.scalar_prod(theta, theta)
step = opt.minimize(direction * cost + tf.maximum(1000.0, norm_theta), var_list=theta) #, global_step=global_step)



from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')#, one_hot=True)


sess = tf.Session()

N = sum([numpy.prod(sh) for sh in sess.run([tf.shape(x) for x in theta])])
print(N)


def converge(epochs = 30, batch_size  = 64, direc = 1.0):
    sess.run(tf.assign(global_step, 0))
    for t in range(epochs):
        x, y = batch(batch_size)
        _, err = sess.run([step, cost], feed_dict={X:x, Y:y, direction:direc})
    return err

def scalar_prod(vs, ws):
    return sum([numpy.sum(v*w) for (v, w) in zip(vs, ws)])


'''
sess.run(tf.global_variables_initializer())
converge()
theta0 = sess.run(theta)

sess.run(tf.global_variables_initializer())
converge()
theta1 = sess.run(theta)

sess.run(tf.global_variables_initializer())
converge()
theta2 = sess.run(theta)

dxs = [v1 - v0 for (v0, v1) in zip(theta0, theta1)]
dys = [v2 - v0 for (v0, v2) in zip(theta0, theta2)]
sc = scalar_prod(dys, dxs) / scalar_prod(dxs, dxs)
print(sc)
dys = [dy - sc*dx for (dx, dy) in zip(dxs, dys)]


'''
sess.run(tf.global_variables_initializer())
theta0 = sess.run(theta)
sess.run(tf.global_variables_initializer())
dxs = sess.run(theta)
sess.run(tf.global_variables_initializer())
dys = sess.run(theta)


w = 10
surface = numpy.zeros([w, w])
surface_x = numpy.zeros([w, w])
surface_y = numpy.zeros([w, w])

zero_x = scalar_prod(theta0, dxs)
zero_y = scalar_prod(theta0, dys)

for (ix, iy) in [(ix, iy) for ix in range(w) for iy in range(w)]:
    x = ix * 1.0 / (w-1.0) - 0.5
    y = iy * 1.0 / (w-1.0) - 0.5
    #x = x * 0.1
    #y = y * 0.1
    sess.run([tf.assign(v, v0 + dx * x + dy * y) for (v, v0, dx, dy) in zip(theta, theta0, dxs, dys)])
    err = converge(100, 128, -1.0)
    surface[ix, iy] = err
    th = sess.run(theta)
    surface_x[ix, iy] = scalar_prod(th, dxs) - zero_x
    surface_y[ix, iy] = scalar_prod(th, dys) - zero_y
    print(ix, iy, err)

sess.close()

surface_dir = numpy.angle(surface_x + 1j * surface_y) 

surface_color = numpy.zeros([w, w, 3])
surface_color[:, :, 0] = (surface_x - numpy.min(surface_x)) / (numpy.max(surface_x) - numpy.min(surface_x))
surface_color[:, :, 1] = (surface_y - numpy.min(surface_y)) / (numpy.max(surface_y) - numpy.min(surface_y))

import matplotlib.pyplot as P

P.imshow(surface)
P.show()

