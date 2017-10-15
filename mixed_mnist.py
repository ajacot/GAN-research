import numpy
import tensorflow as tf

import network as net

batch_size = 32


X = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])
Y = tf.placeholder(tf.float32, shape=[batch_size, 10])
time = tf.constant(0.0)

K = 10

d0 = 1

d_0 = 200
d_1 = 128
d_2 = 128
d_3 = 64

d1 = 104
d2 = 228
d3 = 328
d4 = 1

learning_rate = 0.004
from_save = False

normalize_G = True
normalize_D = True
epochs = 2000

################# GENERATOR ##########3333
Z = tf.placeholder(tf.float32, shape=[batch_size, d_0])
id_Z = tf.placeholder(tf.int32, shape=[batch_size])
one_hot_Z = tf.one_hot(id_Z, K)

centers = tf.Variable(tf.random_normal([K, d_0], name="D_centers"))
logits = tf.Variable(tf.zeros([K, 1]))
probs = tf.exp(logits)
probs = probs / tf.reduce_sum(probs)

Z = Z + tf.matmul(one_hot_Z, centers)

[Z1], vs_0 = net.affine([Z], d_0, d_1*7*7, "G_full")
Z1 = tf.nn.relu(Z1)
Z1, = net.normalize([Z1], [0])
Z1 = tf.reshape(Z1, [-1, 7, 7, d_1])

[X_gen], vs_1 = net.deconv_net([Z1], [d_1, d_2, d_3, d0], [4, 4, 4], [1, 2, 2], batch_size, [7, 7], "G_conv")

X_gen = tf.nn.sigmoid(X_gen)

generator = [centers, logits] + vs_0 + vs_1

tf.summary.image('generated', X_gen, max_outputs=16)

##################### DISCRIMINATOR #######

[X1, X1_gen], vs0 = net.conv_net([X, X_gen], [d0, d1, d2, d3], [4, 4, 4], [2, 2, 1], "D_conv")
                               
X1 = tf.nn.relu(X1)
X1_gen = tf.nn.relu(X1_gen)
X1, X1_gen = net.normalize([X1, X1_gen], [0, 1, 2])
X1 = tf.reshape(X1, [-1, 7*7*d3])
X1_gen = tf.reshape(X1_gen, [-1, 7*7*d3])

[D_real, D_gen], vs1 = net.affine([X1, X1_gen], 7*7*d3, 1, "D")

D_gen = D_gen * tf.matmul(one_hot_Z, probs)

tf.summary.histogram("D_real", D_real)
tf.summary.histogram("D_gen", D_gen)


discriminator = vs0 + vs1

######################### COSTS #############
'''
D_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.ones_like(D_real), D_real)) + \
         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.zeros_like(D_gen), D_gen))
G_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.ones_like(D_gen), D_gen))

D_cost = tf.reduce_mean(D_real - D_gen + tf.nn.relu(D_gen - D_real - tf.norm(tf.reshape(X - X_gen, [batch_size, -1]), axis=1)))
G_cost = tf.reduce_mean(D_gen)

D_cost = tf.reduce_mean(tf.square(D_real)) + tf.reduce_mean(tf.square(D_gen - 1.0))
G_cost = tf.reduce_mean(tf.square(D_gen))

D_cost = tf.reduce_mean(tf.exp(D_real)) + tf.reduce_mean(tf.exp(-D_gen))
G_cost = tf.reduce_mean(tf.exp(D_gen))
'''

D_cost = tf.reduce_mean(D_real) - tf.reduce_mean(D_gen)
G_cost = tf.reduce_mean(D_gen)


tf.summary.scalar("D_cost", D_cost)
tf.summary.scalar("G_cost", G_cost)

tf.summary.tensor_summary("D_gen", D_gen)
tf.summary.tensor_summary("D_real", D_real)

dist = tf.norm(tf.reshape(X - X_gen, [batch_size, -1]), axis=1)

D_cost += 0.001 * tf.reduce_mean(tf.square(D_real)) + 0.001 * tf.reduce_mean(tf.square(D_gen))
G_cost += 0.001 * tf.reduce_mean(tf.square(D_real)) + 0.001 * tf.reduce_mean(tf.square(D_gen))

#D_cost += 0.001 * tf.reduce_mean(tf.square(tf.nn.relu(D_real - D_gen) / (dist+0.1)))
#G_cost += 0.001 * tf.reduce_mean(tf.square(tf.nn.relu(D_gen - D_real) / (dist+0.1)))

#D_cost += 0.1 * tf.reduce_mean(tf.square(vs1[0]))

opt = tf.train.AdamOptimizer(learning_rate, beta1=0.0)
#opt = tf.train.GradientDescentOptimizer(learning_rate)
step_D = opt.minimize(D_cost, var_list=discriminator)
step_G = opt.minimize(G_cost, var_list=generator)


delta_X = tf.gradients(tf.reduce_mean(D_real), [X])[0]
delta_X_gen = tf.gradients(tf.reduce_mean(D_gen), [X_gen])[0]
tf.summary.image('delta_X', delta_X, max_outputs=16)
tf.summary.image('delta_X_gen', delta_X_gen, max_outputs=16)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('SAVE/train', flush_secs=30)


saver = tf.train.Saver()

init = tf.global_variables_initializer()

import pickle

pick_file = open("SAVE/img_evol.pkl", 'wb')
pick = pickle.Pickler(pick_file)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


sess = tf.Session()
if from_save:
    saver.restore(sess, "SAVE/model_mnist89.ckpt")
else:
    sess.run(init)

batch = mnist.train.next_batch(batch_size)
const_x = numpy.reshape(batch[0], [batch_size, 28, 28, 1])
const_z = numpy.random.uniform(-0.5, 0.5, [batch_size, d_0])
const_id_z = numpy.random.randint(0, K, [batch_size])

for t in range(1000):
    for _ in range(1):
        batch = mnist.train.next_batch(batch_size)
        x = numpy.reshape(batch[0], [batch_size, 28, 28, 1])
        z = numpy.random.uniform(-0.5, 0.5, [batch_size, d_0])
        id_z = numpy.random.randint(0, K, [batch_size])

        #print('calculate')
        #grads = sess.run(gen_gradients, feed_dict={X:x, Z:z})
        #print('calculate')

        _, _, cost = sess.run([step_D, step_G, D_cost], feed_dict={X:x, Z:z, id_Z:id_z, time:t})
        print(cost)

    if t % 5 == 0:
        print("SAVE IMAGE")
        summary, x_gen, d_gen, d_real = sess.run([merged, X_gen, D_real, D_gen], feed_dict={X:const_x, Z:const_z, id_Z:const_id_z})
        pick.dump({'X_gen':x_gen, 'D_gen':d_gen, 'D_real':d_real})
        
        train_writer.add_summary(summary, t)
        train_writer.flush()
        

saver.save(sess, 'SAVE/model.ckpt')
# tensorboard --logdir=SAVE/train --reload_interval=4
pick_file.close()

sess.close()

import matplotlib.pyplot as P
w = 2
f, arr = P.subplots(w, w)
for ix in range(w):
    for iy in range(w):
        arr[ix, iy].imshow(x_gen[iy*w + ix, :, :, 0], cmap='gray')
P.show()

