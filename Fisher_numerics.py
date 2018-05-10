import numpy
import tensorflow as tf

import network
import Fisher
import linalg

d = 1
d1 = 150
d2 = 1
X = tf.placeholder(tf.float32, [None, d])
Y = tf.placeholder(tf.float32, [None, d2])
'''
def soft_relu(x):
    return tf.log(1+tf.exp(x * 5.0)) / 5.0
'''
def soft_relu(x):
    a = 0.1
    return 0.5 * x + a - 0.5*(a - tf.nn.relu(x + 0.5*a)) * (1.0 - tf.nn.relu(0.5 - x / a))


def shift_relu(x):
    return (tf.nn.relu(x + 0.1) - 0.3) * 7.0

[YY], theta = network.affine_net([X], [d, d1, d1*2, d1*2, d2], "NET", False)


dy = YY - Y
cost = 0.5 * tf.reduce_mean(tf.square(YY - Y))

#dy = tf.sigmoid(YY) - Y
#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, Y, YY))

#dy = tf.nn.softmax(YY) - Y
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(None, Y, YY))

opt = tf.train.GradientDescentOptimizer(0.1).minimize(cost, var_list=theta)

batch_size = tf.shape(X)[0]

calc_D = Fisher.derivative([YY], theta)
#calc_H = Fisher.compute_Hessian(cost, theta)
#calc_A = Fisher.compute_Hessian([YY], theta, [tf.stop_gradient(dy) / tf.cast(batch_size, tf.float32)])
dist = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(dy), 1)))

def info(sess, feed_dict):
    D = calc_D(sess, feed_dict)

    v_F, e_D, v_K = numpy.linalg.svd(D, False)

    return {'e_D':e_D, 'v_F':v_F, 'v_K':v_K}
'''
def info(sess, feed_dict):
    D = calc_D(sess, feed_dict)
    A = calc_A(sess, feed_dict)
    H = calc_H(sess, feed_dict)

    e_H, v_H = numpy.linalg.eig(H)
    e_A, v_A = numpy.linalg.eig(A)

    v_F, e_D, v_K = numpy.linalg.svd(D, False)

    cross = numpy.dot(numpy.transpose(v_F), v_H)
    return {'D': D, 'H':H, 'A':A,
            'e_H':e_H, 'v_H':v_H, 
            'e_D':e_D, 'v_F':v_F, 'v_K':v_K,
            'e_A':e_A, 'v_A':v_A,
            #'e_K':e_K, 'v_K':v_K,
            'cross':cross}
'''
'''
def info(sess, feed_dict):
    A = calc_A(sess, feed_dict)

    e_A = numpy.linalg.eigvals(A)
    
    return e_A
'''
epochs = 1


import matplotlib.pyplot as P

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    K = sum([numpy.prod(sh) for sh in sess.run([tf.shape(x) for x in theta])])
    print(K)
    eig_As = numpy.zeros([epochs, K])
    costs = numpy.zeros([epochs])
    dists = numpy.zeros([epochs])
    
    batch_size = 2000
    #x = numpy.random.normal(numpy.zeros([batch_size, d]), 0.35) + numpy.random.randint(2, size=[batch_size, 1])*2.0 - 1.0
    x = numpy.reshape(numpy.linspace(-1.0, 1.0, batch_size), [-1, 1])
    y = numpy.square(x) - 0.5
    #y = numpy.absolute(x) > 0.5
    #y = numpy.concatenate([x < -0.5,
    #                       numpy.absolute(x) < 0.5,
    #                       x > 0.5], 1)
    '''
    alpha = numpy.linspace(0.0, numpy.pi*2.0, batch_size)
    x = numpy.stack([numpy.cos(alpha), numpy.sin(alpha)], 1)
    y = x[:, 0:1]'''
    for e in range(epochs):
        for _ in range(e*3*5):
            err, _ = sess.run([cost, opt], feed_dict={X:x, Y:y})
            print(err)
            
        save = info(sess, {X:x, Y:y})
        '''eig_As[e, :] = save.real
        c, d = sess.run([cost, dist], feed_dict={X:x, Y:y})
        costs[e] = c
        dists[e] = d'''
    
    '''
    x = numpy.random.normal(numpy.zeros([batch_size, d]))
    y = numpy.mean(numpy.square(x) - 0.5, 1, keepdims=True)
    hist_simple = info(sess, {X:x, Y:y})

    centers = numpy.random.normal(numpy.zeros([d, d]), 0.7)
    c = numpy.random.randint(d, size=[batch_size])
    x = (centers[c, :]
         + numpy.random.normal(numpy.zeros([batch_size, d]), 0.3))
    y = numpy.mean(numpy.square(x) - 0.5, 1, keepdims=True)
    hist_mixed = info(sess, {X:x, Y:y})
    '''
    
def plot_range1(pl):
    f, axes = P.subplots(1, len(hist), sharex='col', sharey='row')
    for i in range(len(hist)):
        pl(hist[i], axes[i])
    P.show()

def plot_range(pls):
    f, axes = P.subplots(len(pls), len(hist), sharex='col', sharey='row')
    for i in range(len(hist)):
        for j in range(len(pls)):
            pls[j](hist[i], axes[j, i])
    P.show()
    
fig = P.figure();
ax1 = fig.add_subplot(211);
ax1.plot(x, numpy.transpose(save["v_K"][0:5, :]));
ax1.legend(["0", "1", "2", "3", "4"])

ax2 = fig.add_subplot(212);
ax2.plot(x, numpy.transpose(save["v_K"][200:1001:200, :]));
ax2.legend(["200", "400", "600", "800", "1000"])

P.show()

'''
P.semilogy(hist_simple['e_D'])
P.semilogy(hist_mixed['e_D'])

P.show()
'''

'''
up_to = 0
D = hist[0]["D"]
f, axes = P.subplots(2, 4, sharex='col', sharey='row')
centers = [100, 500, 800]
kernels = [numpy.zeros([batch_size, 8]) for _ in centers]

for i in range(4):
    sh = numpy.prod([dim.value for dim in theta[i*2].shape])
    axes[0, i].plot(numpy.transpose(D[up_to:up_to + sh, :]))    
    for (c, k) in zip(centers, kernels):
        k[:, i*2] = numpy.sum(D[up_to:up_to + sh, :] * D[up_to:up_to + sh, c:c+1], 0) / numpy.sqrt(sh*1.0)
    up_to += sh

    sh = numpy.prod([dim.value for dim in theta[i*2+1].shape])
    axes[1, i].plot(numpy.transpose(D[up_to:up_to + sh, :]))
    for (c, k) in zip(centers, kernels):
        k[:, i*2+1] = numpy.mean(D[up_to:up_to + sh, :] * D[up_to:up_to + sh, c:c+1], 0)  / numpy.sqrt(sh*1.0)
    up_to += sh

P.figure()

P.plot(kernels[1][:, 0::2]);
P.gca().set_prop_cycle(None);
P.plot(kernels[1][:, 1::2], '--');
'''


'''
for i in range(num_eigs):
    P.scatter(x[:, 0] + i*4, x[:, 1], c=dxs[i][:, 0] / (eigs[i] + 0.0001))
'''
'''
P.plot(dists, numpy.sort(eig_As));
P.gca().set_ylim(-0.05, 0.05)
P.show()
'''
