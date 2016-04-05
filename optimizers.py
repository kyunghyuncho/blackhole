import theano
from theano import tensor

def rmsprop(tparams, grads, inps, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), 
                                  name='%s_grad'%k) 
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.), 
                                   name='%s_rgrad'%k) 
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), 
                                    name='%s_rgrad2'%k) 
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, numpy.float32(0.95) * rg + numpy.float32(0.05) * g) 
            for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, numpy.float32(0.95) * rg2 + numpy.float32(0.05) * (g ** 2)) 
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inps, cost, updates=zgup+rgup+rg2up)

    updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir'%k) 
             for k, p in tparams.iteritems()]
    updir_new = [(ud, numpy.float32(0.9) * ud - 
                      numpy.float32(1e-4) * zg / 
                      tensor.sqrt(rg2 - rg ** 2 + numpy.float32(1e-4))) 
                 for ud, zg, rg, rg2 
                 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn 
                in zip(tparams.values(), updir_new)]
    f_update = theano.function([], [], 
                               updates=updir_new+param_up, 
                               on_unused_input='ignore')

    return f_grad_shared, f_update
