import numpy
import tensorflow as tf

############


def fwd_gradients(ys, xs, d_xs):
  """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
  With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
  the vector being pushed forward."""
  v = [tf.placeholder_with_default(tf.zeros_like(y), shape=y.get_shape()) for y in ys] # dummy variable
  #v = [tf.placeholder(tf.float32, shape=y.get_shape()) for y in ys] # dummy variable
  g = tf.gradients(ys, xs, grad_ys=v)
  return tf.gradients(g, v, grad_ys=d_xs)



def linear_Fisher(y, x, a=0.01):
    def F(dxs):
      dys = fwd_gradients(y, x, dxs)
      ddxs = tf.gradients(y, x, dys)
      batch_size = tf.cast(tf.shape(y[0])[0], tf.float32)
      return [ddx * (1.0 - a) / batch_size + a * dx for (ddx, dx) in zip(ddxs, dxs)]
    return F

def sigmoid_Fisher(ys, xs, a=0.01):
    sigmoid_ys = [tf.sigmoid(y) for y in ys]
    def F(dxs):
      dys = fwd_gradients(ys, xs, dxs)
      ddxs = tf.gradients(ys, xs, [dy * (sig_y * (1-sig_y)) for (dy, sig_y) in zip(dys, sigmoid_ys)])
      batch_size = tf.cast(tf.shape(y[0])[0], tf.float32)
      return [ddx * (1.0 - a) / batch_size + a * dx for (ddx, dx) in zip(ddxs, dxs)]
    return F

def softmax_Fisher(ys, xs, a=0.01):
    def F(dxs):
      dys = fwd_gradients(ys, xs, dxs)
      ddxs = tf.gradients(ys, xs, dys - tf.reduce_mean(dys, axis=1))
      batch_size = tf.cast(tf.shape(y[0])[0], tf.float32)
      return [ddx * (1.0 - a) / batch_size + a * dx for (ddx, dx) in zip(ddxs, dxs)]
    return F


def Hessian(y, xs):
    def H(dxs):
      grad_xs = tf.gradients([y], xs)
      return [tf.zeros_like(x) if h==None else h for (h, x) in zip(tf.gradients(grad_xs, xs, dxs), xs)]
    return H



################# full computation ##########

def derivative(sess, ys, xs, feed_dict={}):
    x_shapes = sess.run([tf.shape(x) for x in xs], feed_dict=feed_dict)
    y_shapes = sess.run([tf.shape(y) for y in ys], feed_dict=feed_dict)
    N = sum([numpy.prod(s) for s in x_shapes])
    M = sum([numpy.prod(s) for s in y_shapes])

    D = numpy.zeros([N, M])

    ii = 0
    for (y, sh) in zip(ys, y_shapes):
        print(y.name)
        index = tf.placeholder(tf.int32)
        comp_grad = tf.gradients(tf.reshape(y, [-1])[index], xs)
        for i in range(numpy.prod(sh)):
            #dy = tf.reshape(tf.one_hot(i, numpy.prod(sh)), tf.shape(y))
            grads = sess.run(comp_grad, feed_dict={**feed_dict, index:i})
            D[:, ii] = numpy.concatenate([
                    numpy.reshape(grad, [-1])
                    for grad in grads])
            ii += 1
    return D

def compute_linear_Fisher(sess, ys, xs, feed_dict={}):
    D = derivative(sess, ys, xs, feed_dict)
    return numpy.dot(D, numpy.transpose(D)) / D.shape[1]    

def compute_sigmoid_Fisher(sess, ys, xs, feed_dict={}):
    D = derivative(sess, ys, xs, feed_dict)
    sig_ys = sess.run([tf.sigmoid(y) for y in ys], feed_dict=feed_dict)
    sig = numpy.concatenate([numpy.reshape(sig_y * (1-sig_y), [-1])
                             for sig_y in sig_ys])
    return numpy.dot(D*sig, numpy.transpose(D)) / D.shape[1]    

'''
def compute_softmax_Fisher(sess, ys, xs, feed_dict={}):
    D = derivative(sess, ys, xs, feed_dict)
    return numpy.dot(D, numpy.transpose(D)) / D.shape[1]    
'''

def finite_diff_Hessian(sess, xs, grads, d=0.001, feed_dict={}):
    x0s = sess.run(xs, feed_dict=feed_dict)
    N = sum([x0.size for x0 in x0s])

    H = numpy.zeros([N, N])
    
    ii = 0
    for (x, x0) in zip(xs, x0s):
        print(x.name, sess.run(tf.shape(x)))
        for i in range(x0.size):
            D = 0.5*d*tf.reshape(tf.one_hot(i, x0.size), x0.shape)
            
            sess.run(tf.assign(x, x0 + D), feed_dict=feed_dict)
            G_plus = sess.run(grads, feed_dict=feed_dict)
            
            sess.run(tf.assign(x, x0 - D), feed_dict=feed_dict)
            G_minus = sess.run(grads, feed_dict=feed_dict)
            
            H[ii, :] = numpy.concatenate([
                    numpy.reshape((g_plus - g_minus)/d, [-1])
                    for (g_plus, g_minus) in zip(G_plus, G_minus)])
            ii += 1
            sess.run(tf.assign(x, x0))
    return H

def compute_Hessian(sess, y, xs, feed_dict={}):
    return derivative(sess, tf.gradients(y, xs), xs, feed_dict)


'''
import network as net

bs = 20
n = 2

x = tf.placeholder(tf.float32, shape=[bs, n])
y, vs = net.affine_net([x], [n, 5, 5, 1], "N", False)
c = tf.reduce_mean(tf.square(y))

#e_vs = [[tf.random_normal(tf.shape(v), stddev=0.001) for v in vs] for _ in range(4)]
#H = Hessian(c, vs)
#e_vs, es = power_eig(H, e_vs, 10)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

D = derivative(sess, y, vs
        , feed_dict={x:numpy.random.normal(0.0, numpy.ones([bs, n]))})
'''
'''
H = finite_diff_Hessian(sess, vs, tf.gradients(c, vs)
        , feed_dict={x:numpy.random.normal(0.0, numpy.ones([bs, n]))})
'''
#print(sess.run(es, feed_dict={x:numpy.random.normal(0.0, numpy.ones([bs, n]))}))
#print(sess.run(H(e_vs[0]), feed_dict={x:numpy.random.normal(0.0, numpy.ones([bs, n]))}))

#nat_g, err = natural_gradients(c, y, vs, n=6)
#print(sess.run(err, feed_dict={x:numpy.ones([bs, n])}))

