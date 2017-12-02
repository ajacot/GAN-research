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
    for i in range(len(vs)):
        for j in range(i):
            scij = scalar_prod(vs[i], vs[j])
            vs[i] = [vi - vj * scij for (vi, vj) in zip(vs[i], vs[j])]
        ni = norm(vs[i])
        vs[i] = [vi / (ni+0.000) for vi in vs[i]]
    return vs

def power_eig(T, vs, num_steps=5):
    for t in range(num_steps):
        T_vs = [T(v) for v in vs]
        if t==num_steps-1:
            es = [scalar_prod(v, T_v) for (v, T_v) in zip(vs, T_vs)]
        vs = orthogonalize(T_vs)
    return vs, es


def keep_eigs(T, shapes, n = 5, num_steps = 5, shift = 0.0):
    if shift != 0.0:
        def TT(dxs):
            return [dx * shift + Tdx for (dx, Tdx) in zip(dxs, T(dxs))]
    else:
        TT = T
        
    eig_vecs = [[tf.Variable(tf.random_normal(sh, 0.0, 1.0 / tf.sqrt(tf.cast(tf.reduce_prod(sh), tf.float32))))
                 for sh in shapes]
                for i in range(n)]
    new_eig_vecs, eigs = power_eig(TT, eig_vecs, num_steps)
    eigs = [e - shift for e in eigs]
    return eigs, eig_vecs, [tf.assign(v, new_v)
         for (vs, new_vs) in zip(eig_vecs, new_eig_vecs)
         for (v, new_v) in zip(vs, new_vs)]
