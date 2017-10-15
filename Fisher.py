import numpy
import tensorflow as tf


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
      return [ddx + a * dx for (ddx, dx) in zip(ddxs, dxs)]
    return F

def sigmoid_Fisher(ys, xs, a=0.01):
    def F(dxs):
      dys = fwd_gradients(ys, xs, dxs)
      ddxs = tf.gradients(ys, xs, [dy / (y * (1-y)) for (dy, y) in zip(dys, ys)])
      return [ddx + a * dx for (ddx, dx) in zip(ddxs, dxs)]
    return F


def conjgrad(T, b, x, n=100):
    r = [ib - iT for (ib, iT) in zip(b, T(x))]
    p = r
    rsold = sum([tf.reduce_sum(tf.square(ir)) for ir in r])

    err_hist = [rsold]

    for i in range(n):
        Tp = T(p)
        alpha = rsold / sum([tf.reduce_sum(ip * iTp) for (ip, iTp) in zip(p, Tp)])
        x = [ix + alpha * ip for (ix, ip) in zip(x, p)]
        r = [ir - alpha * iTp for (ir, iTp) in zip(r, Tp)]
        rsnew = sum([tf.reduce_sum(tf.square(ir)) for ir in r])

        p = [ir + (rsnew / rsold) * ip for (ir, ip) in zip(r, p)]
        rsold = rsnew
        err_hist = err_hist + [rsold]

    return x, err_hist

def natural_gradients(c, x, F, n=5):
    dx = tf.gradients(c, x)
    nat_dx, err = conjgrad(F, dx, dx, n)
    
    return zip(nat_dx, x), err


def NaturalGradientOptimizer(learning_rate, c, xs, F, n=3):
    opt = tf.train.AdamOptimizer(learning_rate)

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
n = 6

x = tf.placeholder(tf.float32, shape=[bs, n])
y, vs = net.affine_net([x], [6, 50], "N", False)
c = tf.reduce_mean(y)

nat_g, err = natural_gradients(c, y, vs, n=6)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

print(sess.run(err, feed_dict={x:numpy.ones([bs, n])}))
'''

