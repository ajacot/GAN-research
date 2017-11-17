import numpy
import tensorflow as tf

import network as net
import Fisher
import linalg
import train

batch_size = 64


X = tf.placeholder(tf.float32, shape=[None, 2])
time = tf.constant(0.0)

d_0 = 2
d0 = 2

learning_rate = 0.0005
from_save = False

epochs = 3000

################# GENERATOR ##########3333
Z = tf.placeholder(tf.float32, shape=[None, d_0])

[X_gen], generator = net.affine_net([Z], [2, 50, 50, 50, d0], "G", True)

[D_real, D_gen], discriminator = net.affine_net([X, X_gen], [d0, 50, 50, 50, 1], "D", True)


######################### COSTS #############
'''
D_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.ones_like(D_real), D_real)) + \
         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.zeros_like(D_gen), D_gen))
G_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.ones_like(D_gen), D_gen))
'''

D_cost = tf.reduce_mean(D_real) - tf.reduce_mean(D_gen)
G_cost = tf.reduce_mean(D_gen)


#D_cost += 0.1 * tf.reduce_mean(tf.square(D_real)) + 0.1 * tf.reduce_mean(tf.square(D_gen))
#G_cost += 0.1 * tf.reduce_mean(tf.square(D_real)) + 0.1 * tf.reduce_mean(tf.square(D_gen))



opt = tf.train.GradientDescentOptimizer(learning_rate)

info = []
step = []

grads = tf.gradients(D_cost, discriminator) + tf.gradients(G_cost, generator)

### Natural gradient
full_F = Fisher.linear_Fisher([D_real, D_gen], discriminator + generator, 0.01)
grads, err = linalg.conjgrad(full_F, grads, grads, 10)
info = info + [err]

### add damping
#grads, upd = train.damp_gradients(grads)
#step = step + [upd]


step = step + [opt.apply_gradients(zip(grads, discriminator + generator))]



def sample(n):
    theta = numpy.random.uniform(0.0, 2*numpy.pi, [n, 1])
    r = numpy.random.uniform(0.9, 1.1, [n, 1])
    return numpy.concatenate([numpy.sin(theta), numpy.cos(theta)], 1) * r

const_x = sample(batch_size)
grid_x0, grid_x1 = numpy.meshgrid(numpy.linspace(-1.5, 1.5, 16), numpy.linspace(-1.5, 1.5, 16), indexing='ij')
grid_x = numpy.concatenate([numpy.reshape(grid_x0, [-1, 1]), numpy.reshape(grid_x1, [-1, 1])], 1)
const_z = numpy.random.uniform(-0.5, 0.5, [batch_size, d_0])

x_gen = numpy.zeros([0, 2])
running = True
x = sample(batch_size)
d_surf = numpy.zeros([16, 16]) + 1.0

import matplotlib.pyplot as P
import threading

def run_plot():
    P.figure()
    P.xlim([-1.5, 1.5])
    P.ylim([-1.5, 1.5])
    ln_surf = P.imshow(d_surf, animated=True,
                       extent=(-1.5, 1.5, -1.5, 1.5),
                       vmin=-5.0, vmax=5.0,
                       interpolation='bilinear')
    ln_gen, = P.plot(x_gen[:, 0], x_gen[:, 1], 'bo')
    ln_real, = P.plot(const_x[:, 0], const_x[:, 1], 'ro')
    P.ion()
    P.show()
    
    while(running):
        P.pause(0.01)
        ln_surf.set_array(d_surf)
        ln_gen.set_xdata(x_gen[:, 0])
        ln_gen.set_ydata(x_gen[:, 1])
        #ln_real.set_xdata(x[:, 0])
        #ln_real.set_ydata(x[:, 1])
        P.draw()       


thread = threading.Thread(target=run_plot)
thread.deamon = True
thread.start()


sess = tf.Session()

for t in range(epochs):
    for _ in range(1):
        x = const_x # sample(batch_size)
        z = const_z # numpy.random.uniform(-0.5, 0.5, [batch_size, d_0])
            
        if t==0:
            sess.run(tf.global_variables_initializer(), feed_dict={X:x, Z:z})
            print("init")
        
        _, cost, e = sess.run([step, D_cost, info], feed_dict={X:x, Z:z, time:t})
        print(cost)

    if t % 5 == 0:
        x_gen, d_real, d_gen = sess.run([X_gen, D_real, D_gen], feed_dict={X:grid_x, Z:const_z})
        d_surf = numpy.reshape(d_real, [16, 16])
        
running = False
sess.close()


