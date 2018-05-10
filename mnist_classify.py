import numpy
import tensorflow as tf

import network as net
import Fisher
import linalg
import datasets

# 5 => 0.007
# 10 => 0.0045


global_step = tf.Variable(0, trainable=False)

batch_size = 256*2
learning_rate = 0.1

def soft_relu(x):
    a = 0.05
    return 0.5 * x + a - 0.5*(a - tf.nn.relu(x + 0.5*a)) * (1.0 - tf.nn.relu(0.5 - x / a))

def shift_relu(x):
    return (tf.nn.relu(x + 0.3) - 0.7)

X, Y, batch = datasets.mnist(True)

mult = 5
[X1], vs0 = net.conv_net([X], [1, 16*mult, 32*mult, 50*mult], [3, 3, 3], [2, 2, 2],
                         "D_conv", False)
X1 = tf.reshape(soft_relu(X1), [-1, 50*mult*4*4])
#[X1] = net.normalize([X1], [0])

[YY], vs1 = net.affine_net([X1], [50*4*4*mult, 50*4*4*mult, 10], "D_last", soft_relu)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(None, Y, YY))

theta = vs0 + vs1

#opt = tf.train.AdamOptimizer(learning_rate)
opt = tf.train.GradientDescentOptimizer(learning_rate)
grads = tf.gradients(cost, theta)
step = [opt.apply_gradients(zip(grads, theta), global_step=global_step)]

'''
############ calculate F, A, H
ppy = tf.nn.softmax(YY)
dist = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(ppy - Y), 1)))

#F = Fisher.softmax_Fisher([YY], theta)
grad_fixed = tf.gradients(YY, theta, tf.stop_gradient(ppy - Y) / tf.cast(batch_size, tf.float32))
def difference(dxs):
    return [tf.zeros_like(x) if h==None else h
              for (h, x) in zip(tf.gradients(grad_fixed, theta, dxs), theta)]
    #return [Hdx - Fdx for (dx, Fdx, Hdx) in zip(dxs, F(dxs), Hess(dxs))]
#Hess = Fisher.Hessian(cost, theta)

theta_shapes = [tf.shape(v) for v in theta]
num_eigs = 1
step_eigs = []

eigs_F, eig_vecs_F, step_eig = linalg.keep_eigs(F, theta_shapes, num_eigs, 2)
step_eigs = step_eigs + step_eig

eigs_small, eig_vecs_small, step_eig = linalg.keep_eigs(difference, theta_shapes, num_eigs, 2, -1.0)
step_eigs = step_eigs + step_eig
eigs_big, eig_vecs_big, step_eig = linalg.keep_eigs(difference, theta_shapes, num_eigs, 2, 1.0)
step_eigs = step_eigs + step_eig
'''
#dx_F = [Fisher.fwd_gradients([YY], theta, ev)[0] for ev in eig_vecs_F]
#dx_small = [Fisher.fwd_gradients([YY], theta, ev)[0] for ev in eig_vecs_small]
#dx_big = [Fisher.fwd_gradients([YY], theta, ev)[0] for ev in eig_vecs_big]


################# Kernel ######
d0 = tf.gradients(YY[100, 0], theta)
[K0] = Fisher.fwd_gradients([YY], theta, d0)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 10
'''
const_batch = batch(batch_size)

to_save = ([cost, dist] + eigs_small + eigs_big)
hist = {v : numpy.zeros([epochs, numpy.prod(sess.run(tf.shape(v), feed_dict=const_batch))])
        for v in to_save}

N = sum([numpy.prod(sh) for sh in sess.run([tf.shape(x) for x in theta])])
print(N)


for _ in range(50):
    eig, _ = sess.run([eigs_big, step_eigs], feed_dict=const_batch)
    print(eig)
'''

'''
dx = sess.run(dx_F, feed_dict=const_batch)
dx = [d * numpy.sqrt(batch_size) / numpy.linalg.norm(d) for d in dx]
'''
theta0 = sess.run(theta)
dist0 = numpy.sqrt(sum([numpy.sum(numpy.square(v0)) for v0 in theta0]))

kernels = numpy.zeros([epochs, batch_size, 10])
dists = numpy.zeros([epochs])
const_batch = batch(2)
interpolate = numpy.linspace(-0.5, 0.5, batch_size)
interpolate = numpy.reshape(interpolate, [-1, 1, 1, 1])
const_x = const_batch[X][0:1] * interpolate + const_batch[X][1:2]*(1.0-interpolate)
for t in range(epochs):
    for _ in range(50):
        err, _ = sess.run([cost, step], feed_dict=batch(batch_size))
        print(err)
        
    theta1 = sess.run(theta)
    dists[t] = numpy.sqrt(sum([numpy.sum(numpy.square(v0 - v1)) for (v0, v1) in zip(theta0, theta1)])) / dist0
    kernels[t, :, :] = sess.run(K0, feed_dict={X:const_x})
    '''
    for _ in range(2):
        sess.run(step_eigs, feed_dict=const_batch)
    saved = sess.run(to_save, feed_dict=const_batch)
    print(saved[0])
    for (v, val) in zip(to_save, saved):
        hist[v][t, :] = numpy.reshape(val, [-1])
    '''

sess.close()

import matplotlib.pyplot as P
'''
#P.scatter(dx[1], dx[2] , c=numpy.reshape(const_batch[Y], [-1, 1]))
for i in range(0, batch_size):
    P.imshow(1-const_batch[X][i, :, :, 0],
             extent=(dx[1][i]-0.1, dx[1][i][0]+0.1,
                    dx[2][i]-0.1, dx[2][i][0]+0.1),
             cmap='gray')

ax = P.gca();ax.set_xlim(numpy.min(dx[1])-0.1, numpy.max(dx[1])+0.1);ax.set_ylim(numpy.min(dx[2])-0.1, numpy.max(dx[2])+0.1);P.show()
'''
'''
#P.loglog(diff_eigs)
P.plot(hist[dist], numpy.concatenate([hist[eig] for eig in eigs_big], 1));
P.plot(hist[dist], -numpy.concatenate([hist[eig] for eig in eigs_small], 1));
P.show()
'''


P.plot(numpy.squeeze(interpolate), numpy.transpose(kernels[:, :, 0]));
P.show()
