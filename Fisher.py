import numpy
import tensorflow as tf

########## algorithmes #########

def scalar_prod(xs, ys):
    return sum([tf.reduce_sum(x*y) for (x, y) in zip(xs, ys)])

def norm(xs):
    return tf.sqrt(scalar_prod(xs, xs))

def conjgrad(T, b, x, n=100):
    r = [ib - iT for (ib, iT) in zip(b, T(x))]
    p = r
    rsold = scalar_prod(r, r)

    err_hist = [rsold]

    for i in range(n):
        Tp = T(p)
        alpha = rsold / scalar_prod(p, Tp)
        x = [ix + alpha * ip for (ix, ip) in zip(x, p)]
        r = [ir - alpha * iTp for (ir, iTp) in zip(r, Tp)]
        rsnew = scalar_prod(r, r)
        
        p = [ir + (rsnew / rsold) * ip for (ir, ip) in zip(r, p)]
        rsold = rsnew
        err_hist = err_hist + [rsold]

    return x, err_hist


def orthogonalize(vs):
    norms = []
    for i in range(len(vs)):
        for j in range(i):
            scij = scalar_prod(vs[i], vs[j])
            vs[i] = [vi - vj * scij for (vi, vj) in zip(vs[i], vs[j])]
        ni = norm(vs[i])
        vs[i] = [vi / (ni+0.001) for vi in vs[i]]
    return vs

def power_eig(T, vs, n=5):
    for t in range(n):
        T_vs = [T(v) for v in vs]
        es = [scalar_prod(v, T_v) for (v, T_v) in zip(vs, T_vs)]
        vs = orthogonalize(T_vs)
    return vs, es

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
      return [ddx * (1.0 - a) + a * dx for (ddx, dx) in zip(ddxs, dxs)]
    return F

def sigmoid_Fisher(ys, xs, a=0.01):
    sigmoid_ys = [tf.sigmoid(y) for y in ys]
    def F(dxs):
      dys = fwd_gradients(ys, xs, dxs)
      ddxs = tf.gradients(ys, xs, [dy * (sig_y * (1-sig_y)) for (dy, sig_y) in zip(dys, sigmoid_ys)])
      return [ddx * (1.0 - a) + a * dx for (ddx, dx) in zip(ddxs, dxs)]
    return F

def Hessian(y, xs):
    def H(dxs):
      grad_xs = tf.gradients([y], xs)
      return [tf.zeros_like(x) if h==None else h for (h, x) in zip(tf.gradients(grad_xs, xs, dxs), xs)]
    return H

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

def naturalize_gradients(dx, dx0, F, n=5):
    nat_dx, err = conjgrad(F, dx, dx0, n)
    
    return nat_dx, err


def natural_gradients(c, x, F, n=5):
    dx = tf.gradients(c, x)
    nat_dx, err = conjgrad(F, dx, dx, n)
    
    return zip(nat_dx, x), err


def NaturalGradientOptimizer(learning_rate, c, xs, F, n=3):
    opt = tf.train.GradientDescentOptimizer(learning_rate)

    dxs = tf.gradients(c, xs)
    
    last_dxs = [tf.Variable(dx) for dx in dxs]

    nat_dxs, err = conjgrad(F, dxs, last_dxs, n)
    
    steps = [opt.apply_gradients(zip(nat_dxs, xs))]
    steps = steps + [tf.assign(last_dx, nat_dx) for (last_dx, nat_dx) in zip(last_dxs, nat_dxs)]
    return steps, err, [last_dx.initializer for last_dx in last_dxs]

'''
def NaturalGradientOptimizer(learning_rate, c, ys, xs, n=3):
    opt = tf.train.AdamOptimizer(learning_rate)
    
    comp = [[tf.Variable(tf.random_normal(tf.shape(x), stddev=0.001)) for x in xs] for i in range(n)]
    scale = tf.Variable(1.0)

    grad = tf.gradients(c, xs)

    nat_grad = [g * scale for g in grad]

    #scaling = []
    steps = []

    for i in range(n):
        next_comp = apply_LMT(ys, xs, comp[i])
        scaling = sum([tf.reduce_sum(next_c*c) for (next_c, c) in zip(next_comp, comp[i])])
        next_comp = [0.5*c + 0.5*next_c for (next_c, c) in zip(next_comp, comp[i])]
        for j in range(i):
            scalar_prod = sum([tf.reduce_sum(next_c*c) for (next_c, c) in zip(next_comp, comp[j])])
            next_comp = [next_c - scalar_prod*c for (next_c, c) in zip(next_comp, comp[j])]

        norm = sum([tf.reduce_sum(tf.square(next_c)) for next_c in next_comp])
        steps = steps + [tf.assign(c, next_c) for (next_c, c) in zip(next_comp, comp[i])]

        comp_times_grad = sum([tf.reduce_sum(next_c*g) for (next_c, g) in zip(next_comp, grad)])
        nat_grad = [nat_g - comp_times_grad * next_c * (1.0 / (scaling+0.001) - scale) for (next_c, nat_g) in zip(next_comp, nat_grad)]

    steps = steps + tf.assign(scale, scaling)

    step = opt.apply_gradients(zip(nat_grad, xs))

    return [step] + steps
'''


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

