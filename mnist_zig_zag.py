import numpy
import tensorflow as tf

import network as net
import Fisher
import linalg
import train

batch_size = 128


X = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])
Y = tf.placeholder(tf.float32, shape=[batch_size, 10])
time = tf.constant(0.0)

d0 = 1

d_0 = 200
d_1 = 128
d_2 = 128
d_3 = 64

d1 = 104
d2 = 228
d3 = 228
d4 = 1


global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.01, global_step, 200.0, 0.8, staircase=True)
#learning_rate = tf.train.inverse_time_decay(0.0005, global_step, 200.0, 0.25, staircase=True)
#learning_rate = 0.002

FROM_SAVE = False

normalize_G = True
normalize_D = True
epochs = 10000

################# GENERATOR ##########3333
Z = tf.placeholder(tf.float32, shape=[batch_size, d_0])

[Z1], vs_0 = net.affine([Z], d_0, d_1*7*7, "G_full")
Z1 = tf.nn.relu(Z1)
Z1, = net.normalize([Z1], [0])
Z1 = tf.reshape(Z1, [-1, 7, 7, d_1])

[X_gen], vs_1 = net.deconv_net([Z1], [d_1, d_1, d_2, d_3, d0], [3, 4, 4, 4], [1, 1, 2, 2], batch_size, [7, 7], "G_conv")

X_gen = tf.nn.sigmoid(X_gen)

generator = vs_0 + vs_1

tf.summary.image('generated', X_gen, max_outputs=16)

##################### DISCRIMINATOR #######

[X1, X1_gen], vs0 = net.conv_net([X, X_gen], [d0, d1, d2, d3, d3], [4, 4, 4, 2], [2, 2, 1, 1], "D_conv")
                               
X1 = tf.nn.relu(X1)
X1_gen = tf.nn.relu(X1_gen)
X1, X1_gen = net.normalize([X1, X1_gen], [0, 1, 2])
X1 = tf.reshape(X1, [-1, 7*7*d3])
X1_gen = tf.reshape(X1_gen, [-1, 7*7*d3])

[D_real, D_gen], vs1 = net.affine([X1, X1_gen], 7*7*d3, 1, "D")
    
tf.summary.histogram("D_real", D_real)
tf.summary.histogram("D_gen", D_gen)


discriminator = vs0 + vs1

######################### COSTS #############

D_cost = tf.reduce_mean(D_real) - tf.reduce_mean(D_gen)
G_cost = tf.reduce_mean(D_gen)


tf.summary.scalar("D_cost", D_cost)
tf.summary.scalar("G_cost", G_cost)

tf.summary.tensor_summary("D_gen", D_gen)
tf.summary.tensor_summary("D_real", D_real)

opt = tf.train.GradientDescentOptimizer(learning_rate)
#opt = tf.train.AdamOptimizer(learning_rate, 0.5)

grad_G = tf.gradients(G_cost, generator)
grad_D = tf.gradients(D_cost, discriminator)

grads = grad_D + grad_G


### Natural gradient
full_F = Fisher.linear_Fisher([D_real, D_gen], discriminator + generator, 0.01)
grads, err = linalg.conjgrad(full_F, grads, grads, 5)
#info = info + [err]


last_grad = [tf.Variable(tf.zeros_like(v)) for v in discriminator + generator]
save_grad = [tf.assign(v, g) for (v, g) in zip(last_grad, grads)]

step_A = [opt.apply_gradients(zip(grads, discriminator + generator), global_step=global_step), save_grad]
step_B = opt.apply_gradients([(g - 0.3 * l_g, x) for (g, l_g, x) in zip(grads, last_grad, discriminator + generator)])

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


save_dir = "MNIST_GAN"

if FROM_SAVE==False:
    import os
    for file in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, file))

import pickle
pick_file = open(save_dir+"/img_evol.pkl", 'wb')
pick = pickle.Pickler(pick_file)


merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(save_dir, flush_secs=5)

saver = tf.train.Saver()

sess = tf.Session()

if FROM_SAVE:
    saver.restore(sess, "SAVE/model.ckpt")
else:
    sess.run(tf.global_variables_initializer())

try:
    batch = mnist.train.next_batch(batch_size)
    const_x = numpy.reshape(batch[0], [batch_size, 28, 28, 1])
    const_z = numpy.random.uniform(-0.5, 0.5, [batch_size, d_0])

    for t in range(epochs):
        for _ in range(1):
            batch = mnist.train.next_batch(batch_size)
            x = numpy.reshape(batch[0], [batch_size, 28, 28, 1])
            z = numpy.random.uniform(-0.5, 0.5, [batch_size, d_0])
            
            _ = sess.run(step_A, feed_dict={X:x, Z:z, time:t})
            '''
            batch = mnist.train.next_batch(batch_size)
            x = numpy.reshape(batch[0], [batch_size, 28, 28, 1])
            z = numpy.random.uniform(-0.5, 0.5, [batch_size, d_0])
            '''
            _, cost = sess.run([step_B, D_cost], feed_dict={X:x, Z:z, time:t})
            print(cost)

        if t % 5 == 0:
            print("SAVE IMAGE")
            summary, x_gen, d_gen, d_real = sess.run([merged, X_gen, D_real, D_gen], feed_dict={X:const_x, Z:const_z})
            pick.dump({'X_gen':x_gen, 'D_gen':d_gen, 'D_real':d_real})
            
            train_writer.add_summary(summary, t)
            train_writer.flush()
            

    saver.save(sess, save_dir + '/model.ckpt')
finally:
    print("closing")
    pick_file.close()
    sess.close()
    train_writer.close()

# tensorboard --logdir=MNIST_GAN --reload_interval=4

import matplotlib.pyplot as P
w = 2
f, arr = P.subplots(w, w)
for ix in range(w):
    for iy in range(w):
        arr[ix, iy].imshow(x_gen[iy*w + ix, :, :, 0], cmap='gray')
P.show()

'''


saver.restore(sess, "SAVE/model.ckpt")

z0 = numpy.random.uniform(-0.5, 0.5, [d_0])
z1 = numpy.random.uniform(-0.5, 0.5, [d_0])

w = 8
h = 4

z = numpy.stack([z0 * numpy.sin(t) + numpy.cos(t) * z1 for t in numpy.linspace(0, numpy.pi, w*h)])

x_gen = sess.run(X_gen, feed_dict={Z:z})

import matplotlib.pyplot as P
f, arr = P.subplots(h, w)
for ix in range(w):
    for iy in range(h):
        arr[iy, ix].imshow(x_gen[iy*w + ix, :, :, 0], cmap='gray')
P.show()
'''
