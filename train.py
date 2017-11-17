import numpy
import tensorflow as tf

import Fisher
import linalg


def natural_gradients(c, x, F, n=5):
    dx = tf.gradients(c, x)
    nat_dx, err = linalg.conjgrad(F, dx, dx, n)
    
    return zip(nat_dx, x), err


def damp_gradients(dxs, damping=0.001, e=0.01):
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    update_averages = ema.apply(dxs)
    
    mom = [ema.average(x) for x in dxs]
    damp = damping * linalg.scalar_prod(mom, dxs) / (linalg.norm(mom_D)+e)
    return [g - m*damp for (g, m) in zip(dxs, mom)], update_averages


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

