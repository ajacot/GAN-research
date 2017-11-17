import numpy
import tensorflow as tf

import network as net
import Fisher

global_step = tf.Variable(0, trainable=False)

batch_size = 512
learning_rate = tf.train.exponential_decay(0.1, global_step,
                                           200.0, 0.5, staircase=True)

def soft_relu(x):
    return tf.log(1+tf.exp(x))


X = tf.placeholder(tf.float32, shape=[None, 14, 14, 1])
Y = tf.placeholder(tf.float32, shape=[None])

[X1], vs0 = net.conv_net([X], [1, 6, 4], [3, 2], [2, 2], "D_conv", False, soft_relu)
X1 = soft_relu(tf.reshape(X1, [-1, 4*4*4]))
#[X1] = net.normalize([X1])

[YY], vs1 = net.affine([X1], 4*4*4, 1, "D_last")
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.reshape(Y, [-1, 1]), YY))

theta = vs0 + vs1

opt = tf.train.AdamOptimizer(learning_rate)
step = opt.minimize(cost, var_list=theta, global_step=global_step)

'''
grad = tf.gradients(cost, theta)

F = Fisher.sigmoid_Fisher([YY], theta)
H = Fisher.Hessian([cost], theta)

def normed_H(dx):
    Hdx = H(dx)
    FHdx, _ = Fisher.conjgrad(F, Hdx, Hdx, 4)
    return FHdx

k = 2
vs = [[tf.Variable(tf.random_normal(tf.shape(v), stddev=0.001)) for v in theta] for _ in range(k)]

e_vs, es = Fisher.power_eig(normed_H, vs)

step_e = [v.assign(e_v)
          for (vi, e_vi) in zip(vs, e_vs)
          for (v, e_v) in zip(vi, e_vi)]
'''
'''
#scalings = [tf.Variable(1.0) for _ in theta]

nat_grad, nat_err = Fisher.conjgrad(F, grad, grad, 0)
#step_scalings = [tf.assign(s, s*0.99 + 0.01 * tf.norm(nat_g) / tf.norm(g))
#                 for (g, nat_g, s) in zip(grad, nat_grad, scalings)]
'''
#step = opt.apply_gradients(zip(grad, theta), global_step=global_step)


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')#, one_hot=True)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 20
eig_Fs = []
eig_Hs = []
eig_normed_Hs = []
eig_diffs = []
costs = []

N = sum([numpy.prod(sh) for sh in sess.run([tf.shape(x) for x in theta])])
print(N)

batch = mnist.train.next_batch(batch_size)
x = numpy.reshape(batch[0], [batch_size, 28, 28, 1])
x = x[:, 0:28:2, 0:28:2, :]
y = batch[1] == 0
        

for t in range(epochs):
    
    H = numpy.zeros([N, N])
    F = numpy.zeros([N, N])
    n_batches = 1
    for t in range(n_batches):
        print(t)
        '''
        batch = mnist.train.next_batch(batch_size)
        x = numpy.reshape(batch[0], [batch_size, 28, 28, 1])
        x = x[:, 0:28:2, 0:28:2, :]
        y = batch[1] == 0
        ''' 
        print("compute H and F")
        H += Fisher.compute_Hessian(sess, cost, theta, feed_dict={X:x, Y:y})
        
        F += Fisher.compute_sigmoid_Fisher(sess, [YY], theta, feed_dict={X:x, Y:y}) * (1.0 / n_batches)

    print("compute normed_H")
    
    val_F, vec_F = numpy.linalg.eigh(F)
    normed_H = vec_F / numpy.sqrt(numpy.maximum(val_F, 0.00000001))
    normed_H = numpy.linalg.multi_dot([numpy.transpose(normed_H), H, normed_H])

    print("compute eigenvalues")
    eig_Fs = eig_Fs + [val_F]
    eig_Hs = eig_Hs + [numpy.linalg.eigvalsh(H)]
    eig_normed_Hs = eig_normed_Hs + [numpy.linalg.eigvalsh(normed_H)]
    eig_diffs = eig_diffs + [numpy.linalg.eigvalsh(H - F)]
    
    for _ in range(epochs * 3 + 1):
        
        _, err = sess.run([step, cost], feed_dict={X:x, Y:y})
        print(err)
    costs = costs + [err]
    

sess.close()


import matplotlib.pyplot as P


P.plot(costs, numpy.stack(eig_normed_Hs));P.show()
