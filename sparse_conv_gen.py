import tensorflow as tf
import numpy
import network as net

rot90 = numpy.matrix([[0.0, 1.0], [-1.0, 0.0]], numpy.float32)

def convolve(scenes, n, m, d_0, d_1, non_lin=tf.nn.relu):
    vs = [tf.reshape(sc["vs"], [-1, d_0]) for sc in scenes]
    ps = [sc["ps"] for sc in scenes]
    rs = [sc["rs"] for sc in scenes]
    r0 = [tf.reshape(r[:, :, 0], [-1, n, 1, 1]) for r in rs]
    r1 = [tf.reshape(r[:, :, 1], [-1, n, 1, 1]) for r in rs]
    
    vvs, theta0 = net.affine(vs, d_0, m*d_1)
    vvs = [tf.reshape(non_lin(vv), [-1, n*m, d_1]) for vv in vvs]

    pps, theta1 = net.affine(vs, d_0, m*2, var=0.7)
    pps = tf.reshape(pps, [-1, n, m, 2]) * 1.5
    
    pps = pps - tf.reduce_mean(pps, 2, keepdims=True)
    pps = pps / (tf.sqrt(tf.reduce_sum(tf.square(pps), 2, keepdims=True))+0.1)
    
    pps = pps * r0 + tf.reshape(tf.matmul(tf.reshape(pps, [-1, 2]), rot90), [-1, n, m, 2]) * r1
    pps = pps + tf.reshape(ps, [-1, n, 1, 2])
    pps = tf.reshape(pps, [-1, n*m, 2])

    [rrs], theta2 = net.affine([vs], d_0, m*2, var=0.5)
    rrs = tf.reshape(rrs, [-1, n, m, 2])
    rrs = rrs*0.5 + [0.9, 0]
    rrs = 1.0*rrs / (0.3 + tf.norm(rrs, axis=-1, keepdims=True))
    rrs = rrs * r0 + tf.reshape(tf.matmul(tf.reshape(pps, [-1, 2]), rot90), [-1, n, m, 2]) * r1
    rrs = tf.reshape(rrs, [-1, n*m, 2])
    
    return ({"vs":vvs, "ps":pps, "rs":rrs}, theta0+theta1+theta2)





def render(scene, w, sc, n, d_0):
    vs = scene["vs"]
    ps = scene["ps"]
    rs = scene["rs"]
    ns = tf.norm(rs, axis=2, keepdims=True)
    '''
    batch_size = tf.shape(vs)[0]
    
    nw = 7
    ww = 4

    grid_y, grid_x = tf.meshgrid(tf.linspace(0.0, (ww-1)*sc, ww),
                                 tf.linspace(0.0, (ww-1)*sc, ww))
    grid = tf.stack([grid_x, grid_y], axis=2)

    ips = tf.cast(ps / (sc*ww) + nw * 0.5 + 0.5, tf.int32)
    ips = tf.maximum(ips, 0)
    ips = tf.minimum(ips, nw)
    ips = tf.reshape(ips, [-1, 2])
    
    info = tf.concat([vs,
                      ps,
                      ns,
                      tf.one_hot(tf.tile(tf.reshape(tf.range(0, batch_size), [-1, 1]), [1, n]), batch_size)]
                     , 2)
    info = tf.reshape(info, [batch_size*n, -1])
    
    info_split = [tf.dynamic_partition(part, iy, nw+1)
                  for (iy, part) in zip(tf.dynamic_partition(ips[:, 1], ips[:, 0], nw+1),
                                        tf.dynamic_partition(info, ips[:, 0], nw+1))]
    
    def draw_square(ix, iy):
        artis = tf.concat([info_split[ix][iy], info_split[ix][iy+1],
                           info_split[ix+1][iy], info_split[ix+1][iy+1]], 0)
        v, p, n, i_batch = tf.split(artis, [d_0, 2, 1, batch_size], axis=1)
        y = grid - (nw*0.5 - tf.cast([ix, iy], tf.float32))*ww*sc
        y = y - tf.reshape(p, [-1, 1, 1, 2])
        y = y / tf.reshape(n, [-1, 1, 1, 1])
        y = tf.sigmoid(0.3 - 3*tf.reduce_sum(tf.square(y), -1, keep_dims=True))
        y = y * tf.reshape(v * 0.3 + 0.5, [-1, 1, 1, d_0])
        y = tf.tensordot(i_batch, y, [[0], [0]])
        return y
    
    ys = tf.concat([
        tf.concat([
            draw_square(ix, iy)
            for iy in range(nw)], 2)
        for ix in range(nw)], 1)

    
    return ys

    '''
    
    grid_x, grid_y = tf.meshgrid(tf.linspace(-(w-1)*0.5*sc, (w-1)*0.5*sc, w),
                                 tf.linspace(-(w-1)*0.5*sc, (w-1)*0.5*sc, w))
    grid = tf.stack([grid_x, grid_y], axis=2)

    xs = tf.reshape(grid, [1, -1, 1, 2]) - tf.reshape(ps, [-1, 1, n, 2])
    xs = xs / tf.reshape(ns, [-1, 1, n, 1])
    ys = tf.sigmoid(0.3 - tf.reduce_sum(tf.square(xs), -1) * 3.0)
    ys = tf.reshape(ys, [-1, w*w, n, 1]) * tf.reshape(vs, [-1, 1, n, d_0])
    return tf.reshape(tf.reduce_sum(ys, -2), [-1, w, w, d_0])
    
'''
    ns = tf.reshape(tf.reduce_sum(tf.square(rs), 2), [-1, 1, n, 1])
    r0 = tf.reshape(rs[:, :, 0], [-1, 1, n, 1]) / ns
    r1 = -tf.reshape(rs[:, :, 1], [-1, 1, n, 1]) / ns
'''
def identity(x):
    return x

def sparse_net(scene, w, sc, dims, mults, apply_norm=False, non_lin=tf.nn.relu, mod_scene=identity, out_dim=None):
    if out_dim == None:
        out_dim = dims[-1]
        
    thetas = []
    n = 1
    for i in range(len(mults)):
        scene, theta = convolve(scene, n, mults[i], dims[i], dims[i+1], non_lin=non_lin)
        thetas = thetas + theta
        n = n * mults[i]
        if apply_norm:
            scene["vs"] = net.normalize([scene["vs"]], [0, 1])[0]
    scene = mod_scene(scene)
    ys = render(scene, w, sc, n, out_dim)
    return ys, thetas


    
    
