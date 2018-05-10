import tensorflow as tf
import numpy
import sparse_conv_gen as sparse
import datasets

import network as net

d_0 = 100

learning_rate = 5.0

FROM_SAVE = False

normalize_E = False
normalize_G = False
normalize_D = False

epochs = 10000

n = 2
m = 2

batch_size = 32

X, Y, x_batch = datasets.mnist()
Z, z_batch = datasets.random_normal(d_0)

def batch(batch_size):
    return {**x_batch(batch_size), **z_batch(batch_size)}


non_lin_G = net.shift_relu
non_lin_D = net.shift_relu
non_lin_E = net.shift_relu

##################### ENCODER #######

[X1], vs0 = net.conv_net([X*1.5 - 0.5], [1, 104*m, 228*m, 428*m, 428*m, 400*m],
                                 [4, 4, 4, 3, 1], [2, 2, 1, 1, 1],
                                 "E_conv", normalize_E, non_lin_E)
                               
X1 = non_lin_E(X1)
if normalize_E:
    X1 = net.normalize([X1], [0, 1, 2])
X1 = tf.reshape(X1, [-1, 7*7*400*m])

[Y], vs1 = net.affine([X1], 7*7*400*m, d_0, "E")

encoder = vs0 + vs1

tf.summary.histogram("Y_norm", tf.norm(Y, axis=1))

################# GENERATOR ##########3333

[Y_gen], generator_in = net.affine_net([Z], [d_0, d_0 * 3 * n, , d_0 * 3 * n, d_0], "G_in",
                                      normalize_G, non_lin_G)

[Y1, Y1_gen], theta = net.affine([Y, Y_gen], d_0, 500*n)
Y1 = non_lin_G(Y1)
Y1_gen = non_lin_G(Y1_gen)
if normalize_G:
    [Y1] = net.normalize([Y1])
    [Y1_gen] = net.normalize([Y1_gen])

[V0, V0_gen], theta1 = net.affine([Y1, Y1_gen], 500*n, 500*n)
V0 = non_lin_G(V0)
V0_gen = non_lin_G(V0_gen)
if normalize_G:
    [V0_gen] = net.normalize([V0_gen])

[P0, P0_gen], theta2 = net.affine([Y1, Y1_gen], 500*n, 2)
[R0, R0_gen], theta3 = net.affine([Y1, Y1_gen], 500*n, 2)
R0 = tf.reshape(R0, [-1, 1, 2]) + [0.8, 0]
R0_gen = tf.reshape(R0_gen, [-1, 1, 2]) + [0.8, 0]
R0 = 0.8 * R0 / (0.3 + tf.norm(R0, axis=-1, keepdims=True))
R0_gen = 0.8 * R0_gen / (0.3 + tf.norm(R0_gen, axis=-1, keepdims=True))

scene0 = {
    "vs": tf.reshape(V0, [-1, 1, 500*n]),
    "ps": tf.reshape(P0, [-1, 1, 2])*0.3,
    "rs": R0
    }

scene0_gen = {
    "vs": tf.reshape(V0_gen, [-1, 1, 500*n]),
    "ps": tf.reshape(P0_gen, [-1, 1, 2])*0.3,
    "rs": R0_gen
    }

def mod_scene(sc):
    [vs0, vs1, vs2, vs3] = tf.split(sc['vs']*0.3 + 0.5, 4, 1)
    sc['vs'] = tf.concat([
        vs0 * [1, 0, 0],
        vs1 * [0, 1, 0],
        vs2 * [0.5, 0, 0.5],
        vs3 * [0.5, 0.5, 0]
        ], 1)
    return sc

X_gen, generator = sparse.sparse_net(scene0, 28, 3.0 / 28.0,
                [500*n, 400*n, 300*n, 200*n, 100*n, 50*n, 1], [2, 2, 2, 2, 2, 2],
                normalize_G, non_lin=non_lin_G,
                mod_scene = mod_scene, out_dim=3)

tf.summary.image('generated', X_gen, max_outputs=16)
X_gen = tf.reduce_sum(X_gen, 3, keepdims=True)
#X_gen = tf.reshape(X_gen, [-1, 28, 28, 1])
X_gen = tf.atan(tf.nn.relu(X_gen)) * 2 / numpy.pi

generator = generator + theta + theta1 + theta2 + theta3

################ DISCRIMINATOR ######

[X1, X1_gen], ws0 = net.conv_net([X*1.5 - 0.5, X_gen*1.5 - 0.5], [1, 104*m, 228*m, 428*m, 428*m, 400*m],
                                 [4, 4, 4, 3, 1], [2, 2, 1, 1, 1],
                                 "D_conv", normalize_D, non_lin_D)
                               
X1 = non_lin_D(X1)
X1_gen = non_lin_D(X1_gen)
if normalize_D:
    X1 = net.normalize([X1], [0, 1, 2])
    X1_gen = net.normalize([X1_gen], [0, 1, 2])
X1 = tf.reshape(X1, [-1, 7*7*400*m])
X1_gen = tf.reshape(X1_gen, [-1, 7*7*400*m])

[D_real, D_gen], ws1 = net.affine([X1, X1_gen], 7*7*400*m, d_0, "D")

discriminator = ws0 + ws1

tf.summary.histogram("D_real", D_real)
tf.summary.histogram("D_gen", D_gen)


######################### COSTS #############
cost = tf.reduce_mean(tf.square(X - X_gen))


tf.summary.scalar("cost", cost)

opt = tf.train.GradientDescentOptimizer(learning_rate)

step = []

step = opt.minimize(cost, var_list = encoder + generator)

save_dir = "MNIST_AE"

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
    const_dict = batch(batch_size)
    
    for t in range(epochs):
        for _ in range(1):
            _, err = sess.run([step,cost], feed_dict=batch(batch_size))
            #sess.run(fix)
            print(err)

        if t % 5 == 0:
            print("SAVE IMAGE")
            summary, x_gen, err = sess.run([merged, X_gen, cost], feed_dict=const_dict)
            
            train_writer.add_summary(summary, t)
            train_writer.flush()
            

finally:
    print("closing")
    saver.save(sess, save_dir + '/model.ckpt')
    sess.close()
    train_writer.close()

# tensorboard --logdir=MNIST_AE --reload_interval=4


