import numpy
import tensorflow as tf

import network as net
import Fisher

batch_size = 128


X = tf.placeholder(tf.float32, shape=[None, 2])
time = tf.constant(0.0)

d_0 = 2
d0 = 2

learning_rate = 0.003
from_save = False

epochs = 1000

################# GENERATOR ##########3333
Z = tf.placeholder(tf.float32, shape=[None, d_0])

[X_gen], generator = net.affine_net([Z], [2, 50, 50, 50, d0], "G", True)

[D_real, D_gen], discriminator = net.affine_net([X, X_gen], [d0, 50, 50, 50, 1], "D", True)


######################### COSTS #############
'''
D_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.ones_like(D_real), D_real)) + \
         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.zeros_like(D_gen), D_gen))
G_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(None, tf.ones_like(D_gen), D_gen))

D_cost = tf.reduce_mean(D_real - D_gen + tf.nn.relu(D_gen - D_real - tf.norm(tf.reshape(X - X_gen, [batch_size, -1]), axis=1)))
G_cost = tf.reduce_mean(D_gen)

D_cost = tf.reduce_mean(tf.square(D_real)) + tf.reduce_mean(tf.square(D_gen - 1.0))
G_cost = tf.reduce_mean(tf.square(D_gen))


D_cost = tf.reduce_mean(tf.exp(D_real) + D_real) + tf.reduce_mean(tf.exp(-D_gen) - D_gen)
G_cost = tf.reduce_mean(tf.exp(D_gen) + D_gen)
'''

D_cost = tf.reduce_mean(D_real) - tf.reduce_mean(D_gen)
G_cost = tf.reduce_mean(D_gen)


#dist = tf.norm(tf.reshape(X - X_gen, [batch_size, -1]), axis=1)

#D_cost += 0.1 * tf.reduce_sum(tf.square(discriminator[6]))

D_cost += 0.1 * tf.reduce_mean(tf.square(D_real)) + 0.1 * tf.reduce_mean(tf.square(D_gen))
G_cost += 0.1 * tf.reduce_mean(tf.square(D_real)) + 0.1 * tf.reduce_mean(tf.square(D_gen))

#D_cost += 0.001 * tf.reduce_mean(tf.square(D_real - D_gen))
#G_cost += 0.001 * tf.reduce_mean(tf.square(D_real - D_gen))

#D_cost += 0.01 * tf.reduce_mean(tf.square((D_real - D_gen) / (dist+0.5)))
#G_cost += 0.01 * tf.reduce_mean(tf.square((D_gen - D_real) / (dist+0.5)))


#opt = tf.train.AdamOptimizer(learning_rate, beta1=0.0, beta2=0.999)
#opt = tf.train.GradientDescentOptimizer(learning_rate)
#step_D = opt.minimize(D_cost, var_list=discriminator)
#step_G = opt.minimize(G_cost, var_list=generator)

'''
grad_D = opt.compute_gradients(D_cost, var_list=discriminator)
grad_G = opt.compute_gradients(G_cost, var_list=generator)

norm_D = sum([tf.reduce_sum(tf.square(g)) for (g, _) in grad_D])
norm_G = sum([tf.reduce_sum(tf.square(g)) for (g, _) in grad_G])

ema = tf.train.ExponentialMovingAverage(decay=0.999)
step = ema.apply([norm_D, norm_G])

step_D = opt.apply_gradients([(g / tf.sqrt(ema.average(norm_D)+0.01), v)
                              for (g, v) in grad_D])
step_G = opt.apply_gradients([(g / tf.sqrt(ema.average(norm_G)+0.01), v)
                              for (g, v) in grad_G])
'''

'''
### natural gradient with separate Fisher matrices
opt = tf.train.GradientDescentOptimizer(learning_rate)
#F_D = Fisher.sigmoid_Fisher([tf.sigmoid(D_real), tf.sigmoid(D_gen)], discriminator)
F_D = Fisher.linear_Fisher([D_real, D_gen], discriminator)
grad_D, err_D = Fisher.natural_gradients(D_cost, discriminator, F_D, 5)
step_D = opt.apply_gradients(grad_D)

F_G = Fisher.linear_Fisher([D_gen], generator)
#F_G = Fisher.linear_Fisher([X_gen], generator)
grad_G, err_G = Fisher.natural_gradients(G_cost, generator, F_G, 5)
step_G = opt.apply_gradients(grad_G)
'''


## natural gradient with Full Fisher matrix
opt = tf.train.GradientDescentOptimizer(learning_rate)
#scalings = [tf.Variable(1.0) for _ in discriminator + generator]

full_F = Fisher.linear_Fisher([D_real, D_gen], discriminator + generator, 0.01)
grads = tf.gradients(D_cost, discriminator) + tf.gradients(G_cost, generator)
init = tf.global_variables_initializer()
last_grads = [tf.Variable(grad) for grad in grads]


nat_grads, err = Fisher.naturalize_gradients(grads, grads, full_F, 5) 
#        [g*s for (g, s) in zip(grads, scalings)], full_F, 5)
#step_scalings = [tf.assign(s, s*0.99 + 0.01 * tf.norm(nat_g) / tf.norm(g)) for (g, nat_g, s) in zip(grads, nat_grads, scalings)]
step_scalings = [tf.assign(last_grad, last_grad * 0.5 + 0.5 * nat_grad) for (last_grad, nat_grad) in zip(last_grads, nat_grads)]

#step = opt.apply_gradients(zip(nat_grads, discriminator + generator))
step = opt.apply_gradients([(g*0.5 + nat_g*0.5, x)
            for (g, nat_g, x) in zip(grads, nat_grads, discriminator + generator)])

'''
## "inverse" momentum
opt = tf.train.GradientDescentOptimizer(0.0005)
ema = tf.train.ExponentialMovingAverage(decay=0.9)
grad_D = tf.gradients(D_cost, discriminator)
grad_G = tf.gradients(G_cost, generator)
update_averages = ema.apply(grad_D + grad_G)

mom_D = [ema.average(x) for x in grad_D]
mom_G = [ema.average(x) for x in grad_G]
damp_D = 0.00 * Fisher.scalar_prod(mom_D, grad_D) / (Fisher.norm(mom_D)+0.01)
damp_G = 0.00 * Fisher.scalar_prod(mom_G, grad_G) / (Fisher.norm(mom_G)+0.01)
step_D = opt.apply_gradients([(g - m*damp_D, x) for (g, m, x) in zip(grad_D, mom_D, discriminator)]) 
step_G = opt.apply_gradients([(g - m*damp_G, x) for (g, m, x) in zip(grad_G, mom_G, generator)]) 
'''

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
        x = sample(batch_size)
        z = numpy.random.uniform(-0.5, 0.5, [batch_size, d_0])
            
        if t==0:
            #sess.run(tf.global_variables_initializer(), feed_dict={X:x, Z:z})
            sess.run(init, feed_dict={X:x, Z:z})
            sess.run([last_grad.initializer for last_grad in last_grads], feed_dict={X:x, Z:z})
            print("init")
        
        #_, _, cost, eD, eG = sess.run([step_D, step_G, D_cost, err_D, err_G], feed_dict={X:x, Z:z, time:t})
        #_, _, cost = sess.run([step_D, step_G, D_cost], feed_dict={X:x, Z:z, time:t})
        #_, cost, _ = sess.run([step, D_cost, step_scalings], feed_dict={X:x, Z:z, time:t})
        #_, _, cost, _ = sess.run([step_D, step_G, D_cost, update_averages], feed_dict={X:x, Z:z, time:t})
        _, cost, e, _ = sess.run([step, D_cost, err, step_scalings], feed_dict={X:x, Z:z, time:t})
        print(cost)

    if t % 5 == 0:
        #print("SAVE IMAGE")
        x_gen, d_real, d_gen = sess.run([X_gen, D_real, D_gen], feed_dict={X:grid_x, Z:const_z})
        d_surf = numpy.reshape(d_real, [16, 16])
        
running = False
sess.close()
