import tensorflow as tf

########## algorithmes #########

def scalar_prod(xs, ys):
    return sum([tf.reduce_sum(x*y) for (x, y) in zip(xs, ys)])

def norm(xs):
    return tf.sqrt(scalar_prod(xs, xs))

def conjgrad(T, b, x, n=5):
    '''Tx = T(x)
    if scale:
        sc = tf.sqrt(scalar_prod(b, b) / scalar_prod(Tx, Tx))
        x = [xx * sc for xx in x]
        Tx = [Txx * sc for Txx in Tx]'''
    
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
    for i in range(len(vs)):
        for j in range(i):
            scij = scalar_prod(vs[i], vs[j])
            vs[i] = [vi - vj * scij for (vi, vj) in zip(vs[i], vs[j])]
        ni = norm(vs[i])
        vs[i] = [vi / (ni+0.000) for vi in vs[i]]
    return vs

def power_eig(T, vs, num_steps=5, inverse=False, n_conjgrad=1):
    for t in range(num_steps):
        if t > 0:
            orth_vs = orthogonalize(vs)
        else:
            orth_vs = vs
        
        if inverse:
            vs = [conjgrad(T, orth_v, v, n_conjgrad)[0] for (v, orth_v) in zip(vs, orth_vs)]
        else:
            vs = [T(v) for v in orth_vs]
    
    return vs



def keep_eigs(T, shapes, n = 5, num_steps = 5, shift = 0.0, inverse=False, n_conjgrad=1):
    if shift != 0.0:
        def TT(dxs):
            return [dx * shift + Tdx for (dx, Tdx) in zip(dxs, T(dxs))]
    else:
        TT = T
        
    eig_vecs = [[tf.Variable(tf.random_normal(sh, 0.0, 1.0 / tf.sqrt(tf.cast(tf.reduce_prod(sh), tf.float32)))
                             , validate_shape=False)
                 for sh in shapes]
                for i in range(n)]
    new_eig_vecs = power_eig(TT, eig_vecs, num_steps, inverse, n_conjgrad)
    eigs = [norm(v) - tf.abs(shift) for v in new_eig_vecs]
    return eigs, eig_vecs, [tf.assign(v, new_v)
         for (vs, new_vs) in zip(eig_vecs, new_eig_vecs)
         for (v, new_v) in zip(vs, new_vs)]
