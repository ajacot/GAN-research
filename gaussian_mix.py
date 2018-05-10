import numpy
import tensorflow as tf

import network as net
import Fisher
import linalg
import train

batch_size = 64


X = tf.placeholder(tf.float32, shape=[None, 2])
time = tf.constant(0.0)

d_0 = 1
d0 = 2

learning_rate =  0.3 # 0.0001
from_save = False

normalize_G = False
normalize_D = False
non_lin_G = net.shift_relu
non_lin_D = net.shift_relu
mult_b_D = 0.1
mult_b_G = 0.1

epochs = 1000 # 1000

################# GENERATOR ##########3333
Z = tf.placeholder(tf.float32, shape=[None, d_0])

[X_gen], generator = net.affine_net([Z], [d_0, 50, 100, 100, d0], "G"
                                    , apply_norm = normalize_G, non_lin = non_lin_G, mult_b = mult_b_G)

[D_real, D_gen], discriminator = net.affine_net([X, X_gen], [d0, 50, 100, 100, 1], "D"
                                                , apply_norm = normalize_D, non_lin = non_lin_D, mult_b = mult_b_D)


######################### COSTS #############

D_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.ones_like(D_real), D_real)) + \
         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.zeros_like(D_gen), D_gen))
G_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.ones_like(D_gen), D_gen))
'''

D_cost = tf.reduce_mean(D_real) - tf.reduce_mean(D_gen)
G_cost = tf.reduce_mean(D_gen)

'''
#D_cost += 0.1 * tf.reduce_mean(tf.square(D_real)) + 0.1 * tf.reduce_mean(tf.square(D_gen))
#G_cost += 0.1 * tf.reduce_mean(tf.square(D_real)) + 0.1 * tf.reduce_mean(tf.square(D_gen))



opt = tf.train.GradientDescentOptimizer(learning_rate)

info = []
step = []

grads = tf.gradients(D_cost, discriminator) + tf.gradients(G_cost, generator)


step = step + [opt.apply_gradients(zip(grads, discriminator + generator))]


F = Fisher.linear_Fisher([X_gen], generator)
eigs_F, eig_vecs_F, step_eig = linalg.keep_eigs(F, [tf.shape(v) for v in generator], 2, 2)
steepest_grads = [Fisher.fwd_gradients([X_gen], generator, evf)[0] for evf in eig_vecs_F]


step = step + [step_eig]


def sample(n):
    theta = numpy.random.uniform(0.0, 2*numpy.pi, [n, 1])
    r = numpy.random.uniform(0.9, 1.1, [n, 1])
    return numpy.concatenate([numpy.sin(theta), numpy.cos(theta)], 1) * r

const_x = sample(batch_size)
grid_x0, grid_x1 = numpy.meshgrid(numpy.linspace(-1.5, 1.5, 16), numpy.linspace(-1.5, 1.5, 16), indexing='ij')
grid_x = numpy.concatenate([numpy.reshape(grid_x0, [-1, 1]), numpy.reshape(grid_x1, [-1, 1])], 1)
const_z = numpy.random.uniform(-0.5, 0.5, [batch_size, d_0])

x_gen = numpy.zeros([batch_size, 2])
grad_x = numpy.ones([batch_size, 2])
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
    #ln_gen = P.quiver(x_gen[:, 0], x_gen[:, 1] , grad_x[:, 0], grad_x[:, 1])
    ln_real, = P.plot(const_x[:, 0], const_x[:, 1], 'ro')
    P.ion()
    P.show()
    
    while(running):
        P.pause(0.01)
        ln_surf.set_array(d_surf)
        ln_gen.set_xdata(x_gen[:, 0])
        ln_gen.set_ydata(x_gen[:, 1])
        '''
        ln_gen.X = x_gen[:, 0]
        ln_gen.Y = x_gen[:, 1]
        ln_gen.U = grad_x[:, 0]*0.1
        ln_gen.V = grad_x[:, 1]*0.1
        '''
        #ln_real.set_xdata(x[:, 0])
        #ln_real.set_ydata(x[:, 1])
        P.draw()       


thread = threading.Thread(target=run_plot)
thread.deamon = True
thread.start()


sess = tf.Session()

for t in range(epochs):
    for _ in range(1):
        x = sample(batch_size)
        z = numpy.random.uniform(-0.5, 0.5, [batch_size, d_0])
            
        if t==0:
            sess.run(tf.global_variables_initializer(), feed_dict={X:x, Z:z})
            print("init")
        
        _, cost, e = sess.run([step, D_cost, info], feed_dict={X:x, Z:z, time:t})
        print(cost)

    x_gen, grad_x, d_real, d_gen = sess.run([X_gen, steepest_grads[0], D_real, D_gen], feed_dict={X:grid_x, Z:const_z})
    d_surf = numpy.reshape(d_real, [16, 16])
    '''
    if t % 10 == 0:
        x_gen, grad_x, d_real, d_gen = sess.run([X_gen, steepest_grads, D_real, D_gen], feed_dict={X:grid_x, Z:const_z})
        d_surf = numpy.reshape(d_real, [16, 16])

        P.imshow(d_surf, animated=True,
                       extent=(-1.5, 1.5, -1.5, 1.5),
                       vmin=-5.0, vmax=5.0,
                       interpolation='bilinear')
        P.plot(const_x[:, 0], const_x[:, 1], 'bo')
        
        ln_gen = P.quiver(x_gen[:, 0], x_gen[:, 1], grad_x[0][:, 0], grad_x[0][:, 1], color='r')
        ln_gen = P.quiver(x_gen[:, 0], x_gen[:, 1], grad_x[1][:, 0], grad_x[1][:, 1], color='g')

        P.show()
    '''
running = False
sess.close()


