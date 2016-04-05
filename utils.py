import theano
from theano import tensor

import numpy

from collections import OrderedDict()

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzipp(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params

# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')

# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)

def prepare_timeseries(xl, yl=None, max_len=-1):

    lengths = [xx.shape[0] for xx in xl]
    max_length = numpy.max(lengths)

    # TODO: filter out long sequences

    x = numpy.zeros((max_length,len(xl),xl[0].shape[1])).astype('float32')
    x_mask = numpy.zeros((max_length,len(xl))).astype('float32')
    if yl is not None:
        raise Warning('Not supported yet')
    for ii, xx in enumerate(xl):
        x[:len(xx),ii,:] = xx
        x_mask[:len(xx),ii] = 1.

    return x, x_mask
    








