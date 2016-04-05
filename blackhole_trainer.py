import cPickle as pkl
import argparse

import theano
from theano import tensor

import numpy

from optimizers import rmsprop
from utils import *
from layers import *

from blackhole_iter import BHIterator

from collections import OrderedDict


def build_model(options):

    # time x batch x dim
    x = tensor.tensor(name='x', dtype='float32')
    # time x batch
    x_mask = tensor.matrix(name='x_mask', dtype='float32')

    # time x batch x dim
    y = tensor.tensor(name='y', dtype='float32')
    # time x batch
    y_mask = tensor.matrix(name='y_mask', dtype='float32')

    inps = [x, x_mask, y, y_mask]

    opt_rets = OrderedDict()

    # initialize layers
    encoders = []
    decoders = []
    for li in xrange(options['rnn_layers']):
        encoders.append(GRU('encoder%d'%li))
        decoders.append(GRU('decoder%d'%li))
    readout = FC('readout')

    # initialize parameters
    params = OrderedDict()
    nin = options['in_dim']
    for li in xrange(options['rnn_layers']):
        params.update(encoders[li].init_params(nin,options['rnn_units']))
        nin = options['rnn_units']
    for li in xrange(options['rnn_layers']):
        params.update(decoders[li].init_params(nin,options['rnn_units'],cond=True,dim_cond=options['rnn_units']))
    params.update(readout.init_params(options['rnn_units'], options['in_dim']))

    # initialize Theano parameters
    tparams = init_tparams(params)
    for li in xrange(options['rnn_layers']):
        encoders[li].init_tparams(tparams)
        decoders[li].init_tparams(tparams)
    readout.init_tparams(tparams)

    # forward pass
    inp = x
    for li in xrange(options['rnn_layers']):
        rval = encoders[li].execute(inp, mask=x_mask)
        inp = rval[0]
    z = inp[-1]
    opt_ret['z'] = z
    # shift
    inp0 = tensor.alloc(y.shape)
    inp0 = tensor.set_subtensor(inp0[:-1], y[1:])
    inp = inp0
    for li in xrange(options['rnn_layers']):
        rval = decoders[li].execute(inp, mask=y_mask, cond_input=z)
        inp = rval[0]
    rval = readout.execute(inp)
    opt_ret['pred'] = rval

    # cost: mean square error (sum over time)
    cost = (((rval - y) ** 2) * y_mask[:,:,None]).sum(-1).sum(0)

    return params, tparams, cost, inps, opt_ret

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('', '--in-dim', type=int, default=4)
    parser.add_argument('', '--rnn-units', type=int, default=50)
    parser.add_argument('', '--rnn-layers', type=int, default=2)
    parser.add_argument('', '--language-model', action="store_true", default=False)
    parser.add_argument('', '--z-norm', action="store_true", default=False)
    parser.add_argument('', '--n-epochs', type=int, default=2000)
    parser.add_argument('data', type=str)
    parser.add_argument('model', type=str)

    args = parser.parse_args()

    options = dict()
    options['in_dim'] = args.in_dim
    options['rnn_units'] = args.rnn_units
    options['rnn_layers'] = args.rnn_layers
    options['language_model'] = args.language_model
    options['z_norm'] = args.z_norm
    options['n_epochs'] = args.n_epochs

    options['fname'] = args.data
    options['model'] = args.model

    with open('%s.pkl'%args.model, 'wb') as f:
        pkl.dump(options, f)

    # build model
    params, tparams, cost, inps, opt_ret = build_model(options)

    # build optimizer
    grads = tensor.grad(cost.mean(), wrt=tparams)
    f_grad_shared, f_update = rmsprop(tparams, grads, inps, cost)

    trainer = BHIterator(fname=options['data'], batch_size=8, shuffle=True)

    for ee in xrange(options['n_epochs']):
        print 'Epoch', ee
        uidx = 0
        for x, y in trainer:
            uidx += 1
            x_, x_mask = prepare_timeseries(x)

            # TODO: add some noise to the input
            cc = f_grad_shared(x_, x_mask, x_, x_mask)
            f_update()

            print 'Update', uidx, 'Cost', cc

        # pull the latest parameters
        params = unzipp(tparams)
        numpy.savez('%s_iter_%d'%(args.model,ee), params)
        numpy.savez(args.model, params)



if __name__ == '__main__':
    main()
