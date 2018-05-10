import numpy
import tensorflow as tf

import network as net
import Fisher
import linalg
import train
import datasets

#batch_size = 64 * 2




d0 = 1

d_0 = 50
d_1 = 228
d_2 = 228
d_3 = 64

d1 = 104
d2 = 228
d3 = 328
d4 = 1


global_step = tf.Variable(0, trainable=False)
#learning_rate = tf.train.exponential_decay(0.002, global_step, 200.0, 0.8, staircase=True)
#learning_rate = tf.train.inverse_time_decay(0.002, global_step, 200.0, 0.125, staircase=True)
learning_rate = 1.0

FROM_SAVE = True


normalize_G = False
normalize_D = False
non_lin_G = net.shift_relu
non_lin_D = net.shift_relu
mult_b_D = 0.1
mult_b_G = 0.1

epochs = 10000

################# GENERATOR ##########3333

X, Y, x_batch = datasets.mnist()
Z, z_batch = datasets.random_normal(d_0)

def batch(batch_size):
    return {**x_batch(batch_size), **z_batch(batch_size)}

[Z1], vs_0 = net.affine([Z], d_0, d_1*7*7, "G_full", mult_b=mult_b_G)
Z1 = non_lin_G(Z1)
if normalize_G:
    Z1, = net.normalize([Z1], [0])
Z1 = tf.reshape(Z1, [-1, 7, 7, d_1])

tf.summary.scalar('var_z1', tf.reduce_mean(tf.square(Z1)))

[X_gen], vs_1 = net.deconv_net([Z1], [d_1, d_1, d_1, d_2, d_3, d0],
                               [2, 3, 4, 4, 4], [1, 1, 1, 2, 2], tf.shape(Z)[0], [7, 7], "G_conv"
                            , apply_norm = normalize_G, non_lin = non_lin_G, mult_b = mult_b_G)

tf.summary.scalar('var_x_gen', tf.reduce_mean(tf.square(X_gen)))

X_gen = tf.sigmoid(X_gen) + 0.01 * X_gen

generator = vs_0 + vs_1

tf.summary.image('generated', tf.nn.relu(X_gen), max_outputs=16)

##################### DISCRIMINATOR #######

[X1, X1_gen], vs0 = net.conv_net([X*1.5 - 0.5, X_gen * 1.5 - 0.5],
                                 [d0, d1, d2, d3, d3, d3], [4, 4, 4, 3, 1], [2, 2, 1, 1, 1], "D_conv"
                                 , apply_norm = normalize_D, non_lin = non_lin_D, mult_b = mult_b_G, var=1.0)

tf.summary.scalar('var_x1', tf.reduce_mean(tf.square(X1)))

X1 = non_lin_D(X1)
X1_gen = non_lin_D(X1_gen)
if normalize_D:
    X1, X1_gen = net.normalize([X1, X1_gen], [0, 1, 2])
X1 = tf.reshape(X1, [-1, 7*7*d3])
X1_gen = tf.reshape(X1_gen, [-1, 7*7*d3])

[D_real, D_gen], vs1 = net.affine([X1, X1_gen], 7*7*d3, 1, "D", mult_b=mult_b_D, var=0.5)
    
tf.summary.histogram("D_real", D_real)
tf.summary.histogram("D_gen", D_gen)


discriminator = vs0 + vs1

######################### COSTS #############

D_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.ones_like(D_real), D_real)) + \
         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.zeros_like(D_gen), D_gen))
G_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.ones_like(D_gen), D_gen))
'''

D_cost = tf.reduce_mean(D_real) - tf.reduce_mean(D_gen)
G_cost = tf.reduce_mean(D_gen)

D_cost += 0.1 * tf.reduce_mean(tf.square(D_real)) + 0.1 * tf.reduce_mean(tf.square(D_gen))
G_cost += 0.1 * tf.reduce_mean(tf.square(D_real)) + 0.1 * tf.reduce_mean(tf.square(D_gen))
'''
tf.summary.image('gradient', tf.gradients(G_cost, X_gen)[0], max_outputs=4)

tf.summary.scalar("D_cost", D_cost)
tf.summary.scalar("G_cost", G_cost)

tf.summary.tensor_summary("D_gen", D_gen)
tf.summary.tensor_summary("D_real", D_real)


opt = tf.train.GradientDescentOptimizer(learning_rate)
#opt = tf.train.AdamOptimizer(learning_rate, 0.0, 0.999, epsilon=0.0001)

info = []
step = []

grad_G = tf.gradients(G_cost, generator)
grad_D = tf.gradients(D_cost, discriminator)

grads = grad_D + grad_G

step = step + [opt.apply_gradients(zip(grads, discriminator + generator), global_step=global_step)]
step_D = opt.apply_gradients(zip(grad_D, discriminator), global_step=global_step)
step_G = opt.apply_gradients(zip(grad_G, generator))

save_dir = "MNIST_GAN"

if FROM_SAVE==False:
    import os
    for file in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, file))


merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(save_dir, flush_secs=5)

saver = tf.train.Saver()

sess = tf.Session()

if FROM_SAVE:
    saver.restore(sess, save_dir + "/model.ckpt")
else:
    sess.run(tf.global_variables_initializer())

batch_size = 64

try:
    const_dict = batch(batch_size)
    
    for t in range(epochs):
        for _ in range(1):
            #_, cost = sess.run([step, D_cost], feed_dict=batch(batch_size))
            _ = sess.run([step_D], feed_dict=batch(batch_size))
            _, cost = sess.run([step_G, D_cost], feed_dict=batch(batch_size))
            print(cost)

        if t % 5 == 0:
            print("SAVE IMAGE")
            summary, x_gen, d_gen, d_real = sess.run([merged, X_gen, D_real, D_gen], feed_dict=const_dict)
            
            train_writer.add_summary(summary, t)
            train_writer.flush()
            

finally:
    print("closing")
    saver.save(sess, save_dir + '/model.ckpt')
    sess.close()
    train_writer.close()

# tensorboard --logdir=MNIST_GAN --reload_interval=4


'''


saver.restore(sess, save_dir+"/model.ckpt")

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
