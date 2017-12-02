import numpy
import tensorflow as tf

import network as net
import Fisher
import linalg

global_step = tf.Variable(0, trainable=False)

batch_size = 812
learning_rate = tf.train.exponential_decay(0.05, global_step,
                                           200.0, 0.5, staircase=True)

def soft_relu(x):
    return tf.log(1+tf.exp(x))


X = tf.placeholder(tf.float32, shape=[None, 14, 14, 1])
Y = tf.placeholder(tf.float32, shape=[None, 10])

[X1], vs0 = net.conv_net([X], [1, 8, 16, 16], [3, 2, 2], [2, 2, 1], "D_conv", True, soft_relu)
X1 = soft_relu(tf.reshape(X1, [-1, 16*4*4]))
[X1] = net.normalize([X1])

[YY], vs1 = net.affine([X1], 16*4*4, 10, "D_last")
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(None, Y, YY))

ppy = tf.nn.softmax(YY)
pp = tf.reduce_mean(ppy, axis=0)
HH = -tf.reduce_sum(pp * tf.log(pp))

dist = tf.reduce_mean(tf.square(ppy - Y))

p = tf.reduce_mean(Y, axis=0)
H = -tf.reduce_sum(p * tf.log(p))

theta = vs0 + vs1

#opt = tf.train.AdamOptimizer(learning_rate)
opt = tf.train.GradientDescentOptimizer(learning_rate)
grads = tf.gradients(cost, theta)
step = [opt.apply_gradients(zip(grads, theta), global_step=global_step)]

ema = tf.train.ExponentialMovingAverage(decay=0.99)
step = step + [ema.apply(grads)]
intensity = tf.sqrt(sum([tf.reduce_sum(tf.square(ema.average(g))) for g in grads]))
variability = tf.sqrt(sum([tf.reduce_sum(tf.square(g - ema.average(g))) for g in grads]))
theta_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(x)) for x in theta]))


F = Fisher.softmax_Fisher([YY], theta)
Hess = Fisher.Hessian(cost, theta)
grad_fixed = tf.gradients(YY, theta, tf.stop_gradient(ppy - Y))
def difference(dxs):
    return [tf.zeros_like(x) if h==None else h / batch_size
              for (h, x) in zip(tf.gradients(grad_fixed, theta, dxs), theta)]
    #return [Hdx - Fdx for (dx, Fdx, Hdx) in zip(dxs, F(dxs), Hess(dxs))]


theta_shapes = [tf.shape(v) for v in theta]
num_eigs = 4
step_eigs = []
eigs_F, eig_vecs_F, step_eig = linalg.keep_eigs(F, theta_shapes, num_eigs, 2)
step_eigs = step_eigs + step_eig
eigs_small, eig_vecs_small, step_eig = linalg.keep_eigs(difference, theta_shapes, num_eigs, 2, -1.0)
step_eigs = step_eigs + step_eig
eigs_big, eig_vecs_big, step_eig = linalg.keep_eigs(difference, theta_shapes, num_eigs, 2, 1.0)
step_eigs = step_eigs + step_eig

step = step + step_eigs

F_affinity = [linalg.scalar_prod(grads, v) for v in  eig_vecs_F]
small_affinity = [linalg.scalar_prod(grads, v) for v in  eig_vecs_small]
big_affinity = [linalg.scalar_prod(grads, v) for v in eig_vecs_big]

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 500

to_save = ([cost, dist] + eigs_F + eigs_small + eigs_big +
           F_affinity + small_affinity + big_affinity + 
           [intensity, variability, theta_norm,
           H, HH])
hist = {v : numpy.zeros([epochs, numpy.prod(sess.run(tf.shape(v)))])
        for v in to_save}

N = sum([numpy.prod(sh) for sh in sess.run([tf.shape(x) for x in theta])])
print(N)

#calculate_H = Fisher.compute_Hessian(cost, theta)
#calculate_F = Fisher.compute_softmax_Fisher([YY], theta)

for _ in range(3):
    batch = mnist.train.next_batch(batch_size)
    x = numpy.reshape(batch[0], [batch_size, 28, 28, 1])
    x = x[:, 0:28:2, 0:28:2, :]
    y = batch[1]

    sess.run(step_eigs, feed_dict={X:x, Y:y})
    
for t in range(epochs):
    
    batch = mnist.train.next_batch(batch_size)
    x = numpy.reshape(batch[0], [batch_size, 28, 28, 1])
    x = x[:, 0:28:2, 0:28:2, :]
    y = batch[1]

    '''
    print("compute H and F")
    H = calculate_H(sess, feed_dict={X:x, Y:y})
    
    F, D = calculate_F(sess, feed_dict={X:x, Y:y})

    print("compute normed_H")
    
    val_F, vec_F = numpy.linalg.eigh(F)
    normed_H = vec_F / numpy.sqrt(numpy.maximum(val_F, 0.00000001))
    normed_H = numpy.linalg.multi_dot([numpy.transpose(normed_H), H, normed_H])

    print("compute eigenvalues")
    eig_Fs = eig_Fs + [val_F]
    eig_Hs = eig_Hs + [numpy.linalg.eigvalsh(H)]
    eig_normed_Hs = eig_normed_Hs + [numpy.linalg.eigvalsh(normed_H)]
    eig_diffs = eig_diffs + [numpy.linalg.eigvalsh(H - F)]
    '''
    
    (_, saved) = sess.run([step, to_save], feed_dict={X:x, Y:y})
    print(saved[0])
    for (v, val) in zip(to_save, saved):
        hist[v][t, :] = numpy.reshape(val, [-1])

W0 = sess.run(theta[0])

vec_F, vec_small, vec_big = sess.run([eig_vecs_F, eig_vecs_small, eig_vecs_big])

sess.close()


import matplotlib.pyplot as P

#P.plot(costs, numpy.stack(eig_normed_Hs));P.show()

P.loglog(hist[cost] + numpy.log(0.1));
P.loglog(hist[dist]);
P.loglog(hist[intensity]);
P.loglog(hist[variability]);
P.loglog(numpy.concatenate([hist[eig] for eig in eigs_F], 1));
P.figure()

#P.loglog(diff_eigs)
P.semilogx(numpy.concatenate([hist[eig] for eig in eigs_big], 1));
P.semilogx(numpy.concatenate([hist[eig] for eig in eigs_small], 1));
P.show()
'''
P.figure()

P.semilogx(numpy.absolute(small_affinities) + 5);
P.semilogx(numpy.absolute(big_affinities));
'''
P.show()
