import tensorflow as tf
import numpy
import sparse_conv_gen as sparse
import datasets

import network as net

d_0 = 100

learning_rate = 2.0

FROM_SAVE = False

normalize_G = False
normalize_D = False

epochs = 10000

n = 4
m = 3

batch_size = 16

iterator = datasets.celeb_A().batch(batch_size).make_one_shot_iterator()
Z, z_batch = datasets.random_normal(d_0)
X = iterator.get_next()

def batch(batch_size):
    return {**x_batch(batch_size), **z_batch(batch_size)}



non_lin_G = net.shift_relu
non_lin_D = tf.nn.relu

################# GENERATOR ##########3333
'''
xs = tf.lin_space(-1.5, 1.5, 28)
xs = tf.stack([tf.tile(tf.reshape(xs, [-1, 1]), [1, 28]),
               tf.tile(tf.reshape(xs, [1, -1]), [28, 1])], 2)
xs = tf.reshape(xs, [-1, 2])
'''
[Z1], theta = net.affine([Z], d_0, 500*n)
Z1 = non_lin_G(Z1)
if normalize_G:
    [Z1] = net.normalize([Z1])

[V0], theta1 = net.affine([Z1], 500*n, 500*n)
V0 = non_lin_G(V0)
if normalize_G:
    [V0] = net.normalize([V0])

[P0], theta2 = net.affine([Z1], 500*n, 2)
[R0], theta3 = net.affine([Z1], 500*n, 2)
R0 = tf.reshape(R0, [-1, 1, 2]) + [0.8, 0]
R0 = 0.8 * R0 / (0.3 + tf.norm(R0, axis=-1, keepdims=True))

scene0 = {
    "vs": tf.reshape(V0, [-1, 1, 500*n]),
    "ps": tf.reshape(P0, [-1, 1, 2])*0.3,
    "rs": R0
    }


def mod_scene(sc):
    sc['rs'] = sc['rs']*2.0
    return sc


X0_gen, generator = sparse.sparse_net(scene0, 64, 3.0 / 64.0,
                [500*n, 400*n, 300*n, 200*n, 100*n, 50*n, 20*n], [2, 2, 2, 2, 3, 3],
                normalize_G, non_lin=non_lin_G,
                mod_scene=mod_scene)

X0_gen = non_lin_G(X0_gen)

#[X_gen], theta4 = net.affine([tf.reshape(X0_gen, [batch_size*64*64, -1])], 20*n, 3)
#X_gen = tf.reshape(X_gen, [-1, 64, 64, 3])

[W_last], theta_W = net.affine([Z1], 500*n, 20*n*3)
[b_last], theta_b = net.affine([Z1], 500*n, 3)
X_gen = tf.matmul(tf.reshape(X0_gen, [batch_size, 64*64, -1]), tf.reshape(W_last, [batch_size, -1, 3])) / tf.sqrt(20.0*n)
X_gen = X_gen + 0.5 + 0.1 * tf.reshape(b_last, [batch_size, 1, 3])
X_gen = tf.reshape(X_gen, [-1, 64, 64, 3])

X_gen = tf.atan(tf.nn.relu(X_gen)) * 2 / numpy.pi
tf.summary.image('generated', X_gen, max_outputs=16)

generator = generator + theta + theta1 + theta2 + theta3 + theta_W + theta_b

##################### DISCRIMINATOR #######

[X1, X1_gen], vs0 = net.conv_net([X*1.5 - 0.5, X_gen * 1.5 - 0.5], [3, 104*m, 228*m, 428*m, 428*m, 400*m],
                                 [4, 4, 4, 3, 1], [2, 2, 2, 2, 1],
                                 "D_conv", normalize_D, non_lin_D)
                               
X1 = non_lin_D(X1)
X1_gen = non_lin_D(X1_gen)
if normalize_D:
    X1, X1_gen = net.normalize([X1, X1_gen], [0, 1, 2])
X1 = tf.reshape(X1, [-1, 4*4*400*m])
X1_gen = tf.reshape(X1_gen, [-1, 4*4*400*m])

[D_real, D_gen], vs1 = net.affine([X1, X1_gen], 4*4*400*m, 1, "D")
    
tf.summary.histogram("D_real", D_real)
tf.summary.histogram("D_gen", D_gen)


discriminator = vs0 + vs1

######################### COSTS #############
'''
D_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.ones_like(D_real), D_real)) + \
         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.zeros_like(D_gen), D_gen))
G_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.ones_like(D_gen), D_gen))

'''
D_cost = tf.reduce_mean(D_real) - tf.reduce_mean(D_gen)
G_cost = tf.reduce_mean(D_gen)

D_cost += 0.5 * tf.reduce_mean(tf.square(D_real)) + 0.5 * tf.reduce_mean(tf.square(D_gen))
G_cost += 0.5 * tf.reduce_mean(tf.square(D_real)) + 0.5 * tf.reduce_mean(tf.square(D_gen))


tf.summary.scalar("D_cost", D_cost)
tf.summary.scalar("G_cost", G_cost)

tf.summary.tensor_summary("D_gen", D_gen)
tf.summary.tensor_summary("D_real", D_real)



opt = tf.train.GradientDescentOptimizer(learning_rate)

info = []
step = []

grad_G = tf.gradients(G_cost, generator)
grad_D = tf.gradients(D_cost, discriminator)

grads = grad_D + grad_G

step = step + [opt.apply_gradients(zip(grads, discriminator + generator))]
fix = net.fix_weights(generator) + net.fix_weights(discriminator)
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_gen = sess.run(X_gen, feed_dict=batch(2))

import matplotlib.pyplot as P

P.imshow(x_gen[0, :, :, 0]);P.show()

'''
save_dir = "CELEB_GAN"

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


try:
    const_dict = z_batch(batch_size)
    
    for t in range(epochs):
        for _ in range(1):
            _, cost = sess.run([step, D_cost], feed_dict=z_batch(batch_size))
            #sess.run(fix)
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

# tensorboard --logdir=CELEB_GAN --reload_interval=4


