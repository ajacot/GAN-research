import numpy
import tensorflow as tf

import network as net
import Fisher
import linalg
import datasets

# 5 => 0.007
# 10 => 0.0045


global_step = tf.Variable(0, trainable=False)

batch_size = 400
lr = 0.5 #tf.placeholder(tf.float32)

d1 = tf.placeholder(tf.int32)

X, _, batch = datasets.mnist(True)

Y = tf.placeholder(tf.float32, [batch_size, 1])

[YY], theta = net.affine_net([tf.reshape(X, [batch_size, -1])], [28*28, d1, d1, d1, d1, 1], "NET",
                             False, non_lin=net.shift_relu, mult_b=0.1)
'''
mult=5
[X1], vs0 = net.conv_net([X], [1, 16*mult, 32*mult, 50*mult], [3, 3, 3], [2, 2, 2],
                         "D_conv", False)
X1 = tf.reshape(tf.nn.relu(X1), [-1, 50*mult*4*4])
[YY], vs1 = net.affine_net([X1], [50*4*4*mult, 50*4*4*mult, 1], "D_last")
theta = vs0 + vs1
'''
cost = 0.5 * tf.reduce_mean(tf.square(Y - YY))

opt = tf.train.GradientDescentOptimizer(lr)
step = opt.minimize(cost, var_list=theta)

################# Kernel ######
NTK = Fisher.tangent_kernel([YY], theta)
def NTK1(dys):
    return [dy / (batch_size*1.0) for dy in NTK(dys)]

eigs_NTK, eig_vecs_NTK, step_eig = linalg.keep_eigs(NTK1, [Y.shape], 3, 2)


sess = tf.Session()

first_d1 = 10000
sess.run(tf.global_variables_initializer(), feed_dict={d1 : first_d1})

const_batch = batch(batch_size)

for t in range(10):
    print(t)
    sess.run(step_eig, feed_dict={d1 : first_d1, **const_batch})

ex, dx = sess.run([eigs_NTK, eig_vecs_NTK], feed_dict={d1 : first_d1, **const_batch})
df1 = numpy.array(dx[1][0])
df1 = df1 / numpy.sqrt(numpy.mean(numpy.square(df1)))
df2 = numpy.array(dx[2][0])
df2 = df2 / numpy.sqrt(numpy.mean(numpy.square(df2)))

T = 1000
num_sizes = 3
num_tries = 2
conv_in = numpy.zeros([T, num_tries, num_sizes])
conv_out = numpy.zeros([T, num_tries, num_sizes])

for i in range(num_sizes):
    dd1 = [100, 1000, 10000][i]
    for j in range(num_tries):
        sess.run(tf.global_variables_initializer(), feed_dict={d1 : dd1})
        y0 = sess.run(YY, feed_dict={d1 : dd1, **const_batch})
        y = y0 + df1*0.5

        for t in range(T):
            yy, _, err = sess.run([YY, step, cost], feed_dict={d1 : dd1, **const_batch, Y:y})
            print(err)
            conv_in[t, j, i] = numpy.mean((yy - y) * df1)
            conv_out[t, j, i] = numpy.sqrt(numpy.mean(numpy.square(yy - y - conv_in[t, j, i]*df1)))
    
sess.close()

import matplotlib.pyplot as P

def prep_image(img):
    return numpy.concatenate([numpy.zeros([28, 28, 3]), img], 2)

#P.scatter(dx[1], dx[2] , c=numpy.reshape(const_batch[Y], [-1, 1]))
for i in range(0, batch_size):
    P.imshow(prep_image(const_batch[X][i, :, :]),
             extent=(df1[i][0]-0.09, df1[i][0]+0.09,
                    df2[i][0]-0.09, df2[i][0]+0.09))

ax = P.gca();P.xlabel("$f^{(2)}(x)$");P.ylabel("$f^{(3)}(x)$");
ax.set_xlim(numpy.min(df1)-0.1, numpy.max(df1)+0.1);
ax.set_ylim(numpy.min(df2)-0.1, numpy.max(df2)+0.1);

P.figure()
time = numpy.linspace(0, T*lr, T)
P.plot(time, -conv_in[:, 0, 0], "g:", alpha=0.5, label="$n=100$")
P.plot(time, -conv_in[:, 1:, 0], "g:", alpha=0.5)
P.plot(time, -conv_in[:, 0, 1], "r:", alpha=0.5, label="$n=1000$")
P.plot(time, -conv_in[:, 1:, 1], "r:", alpha=0.5)
P.plot(time, -conv_in[:, 0, 2], "b:", alpha=0.5, label="$n=10000$")
P.plot(time, -conv_in[:, 1:, 2], "b:", alpha=0.5)
P.plot(time, 0.5* numpy.exp(-time * ex[1]), label="$n=\infty$")
P.xlim([0, T*lr]);P.xlabel("$t$");P.legend()

P.figure()
P.plot(time, conv_out[:, 0, 0], "g:", alpha=0.5, label="$n=100$")
P.plot(time, conv_out[:, 1:, 0], "g:", alpha=0.5)
P.plot(time, conv_out[:, 0, 1], "r:", alpha=0.5, label="$n=1000$")
P.plot(time, conv_out[:, 1:, 1], "r:", alpha=0.5)
P.plot(time, conv_out[:, 0, 2], "b:", alpha=0.5, label="$n=10000$")
P.plot(time, conv_out[:, 1:, 2], "b:", alpha=0.5)
P.xlim([0, T*lr]);P.xlabel("$t$");P.legend()



P.show()

