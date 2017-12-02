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
    batch_size = tf.cast(tf.shape(dys[0])[0], tf.float32)
    def F(dxs):
      dys = fwd_gradients(ys, xs, dxs)
      ddxs = tf.gradients(ys, xs, [dy * (sig_y * (1-sig_y)) for (dy, sig_y) in zip(dys, sigmoid_ys)])
      return [ddx * (1.0 - a) / batch_size + a * dx for (ddx, dx) in zip(ddxs, dxs)]
    return F

def softmax_Fisher(ys, xs, a=0.01):
    ps = [tf.nn.softmax(y) for y in ys]
    shape_y = tf.shape(ys[0])
    batch_size = tf.cast(shape_y[0], tf.float32)
    num_classes = tf.cast(shape_y[1], tf.float32)
    def F(dxs):
      dys = fwd_gradients(ys, xs, dxs)
      ddxs = tf.gradients(ys, xs, [p*(dy - tf.reduce_sum(dy * p, axis=1, keep_dims=True)) for (dy, p) in zip(dys, ps)])
      return [ddx * (1.0 - a) / batch_size + a * dx for (ddx, dx) in zip(ddxs, dxs)]
    return F


def Hessian(y, xs):
    grad_xs = tf.gradients([y], xs)
    def H(dxs):
      return [tf.zeros_like(x) if h==None else h
              for (h, x) in zip(tf.gradients(grad_xs, xs, dxs), xs)]
    return H



################# full computation ##########

def derivative(ys, xs):
    index = tf.placeholder(tf.int32)
    comp_grad = [tf.gradients(tf.reshape(y, [-1])[index], xs)
                 for y in ys]
    def_y_shapes = [tf.shape(y) for y in ys]
    def_x_shapes = [tf.shape(x) for x in xs]

    def calculate(sess, feed_dict={}):
      y_shapes = sess.run(def_y_shapes, feed_dict=feed_dict)
      x_shapes = sess.run(def_x_shapes, feed_dict=feed_dict)
      
      M = sum([numpy.prod(s) for s in y_shapes])
      N = sum([numpy.prod(s) for s in x_shapes])

      D = numpy.zeros([N, M])
      
      ii = 0
      for (y, c_grad, sh) in zip(ys, comp_grad, y_shapes):
          print(y.name)
          for i in range(numpy.prod(sh)):
              grads = sess.run(c_grad, feed_dict={**feed_dict, index:i})
              D[:, ii] = numpy.concatenate([
                      numpy.reshape(grad, [-1])
                      for grad in grads])
              ii += 1
      return D
    return calculate

def compute_linear_Fisher(ys, xs):
    cal_D = derivative(ys, xs)
    
    def calculate(sess, feed_dict={}):
      D = cal_D(sess, feed_dict)
      return numpy.dot(D, numpy.transpose(D)) / D.shape[1], D
    return calculate

def compute_sigmoid_Fisher(ys, xs):
    cal_D = derivative(ys, xs)
    
    def calculate(sess, feed_dict={}):
      D = cal_D(sess, feed_dict)
      sig_ys = sess.run([tf.sigmoid(y) for y in ys], feed_dict=feed_dict)
      sig = numpy.concatenate([numpy.reshape(sig_y * (1-sig_y), [-1])
                               for sig_y in sig_ys])
      return numpy.dot(D*sig, numpy.transpose(D)) / D.shape[1], D
    return calculate

def compute_softmax_Fisher(ys, xs):
    cal_D = derivative(ys, xs)
    def_sh = tf.shape(ys[0])
    
    def calculate(sess, feed_dict={}):
      D = cal_D(sess, feed_dict)
      sh = sess.run(def_sh, feed_dict)
      
      normed_D = numpy.reshape(D, [-1, sh[0], sh[1]])
      normed_D = normed_D - numpy.mean(normed_D, axis=2, keepdims=True)
      normed_D = numpy.reshape(D, [-1, sh[0] * sh[1]])
      return numpy.dot(normed_D, numpy.transpose(normed_D)) / D.shape[1], D
    return calculate

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

def compute_Hessian(y, xs):
    return derivative(tf.gradients(y, xs), xs)


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

