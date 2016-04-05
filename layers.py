import theano
from theano import tensor

import numpy

from collections import OrderedDict

from utils import *

class Layer:
    def __init__(self, name):
        self.name = name
        self.params = OrderedDict()
        self.tparams = OrderedDict()

    def __param(self, name):
        return self.params[self.__pname__(name)]

    def __tparam(self, name):
        return self.tparams[self.__pname__(name)]

    def __pname(self, name):
        return _p(self.name, name)

    def init_tparams(self, tparams):
        for kk, vv in tparams.iteritems():
            if kk in self.params:
                self.tparams[kk] = vv

    def init_params(self):
        return None

    def execute(self, state_below):
        return None

class FC(Layer):
    def init_params(self, nin, nout, ortho=True):
        self.params[self.__pname('W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
        self.params[self.__pname('b')] = numpy.zeros((nout,)).astype('float32')
        return self.params

    def execute(self, state_below, activ='lambda x: tensor.tanh(x)'):
        return eval(activ)(tensor.dot(state_below, self.__tparams('W'))+self.__tparams('b'))

class GRU(Layer):
    def init_params(self, nin, dim, cond=False, dim_cond=-1):
        # embedding to gates transformation weights, biases
        W = numpy.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim)], axis=1)
        self.params[self.__pname('W')] = W
        self.params[self.__pname('b')] = numpy.zeros((2 * dim,)).astype('float32')

        # recurrent transformation weights for gates
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        self.params[self.__pname('U')] = U

        # embedding to hidden state proposal weights, biases
        Wx = norm_weight(nin, dim)
        self.params[self.__pname('Wx')] = Wx
        self.params[self.__pname('bx')] = numpy.zeros((dim,)).astype('float32')

        # recurrent transformation weights for hidden state proposal
        Ux = ortho_weight(dim)
        self.params[self.__pname('Ux')] = Ux

        self.cond = cond
        if cond:
            assert dim_cond > 0
            # embedding to gates transformation weights, biases
            W_cond = numpy.concatenate([norm_weight(dim_cond, dim),
                                        norm_weight(dim_cond, dim)], axis=1)
            self.params[self.__pname('W_cond')] = W
        
            # embedding to hidden state proposal weights, biases
            Wx = norm_weight(dim_cond, dim)
            self.params[self.__pname('Wx_cond')] = Wx_cond

        return self.params

    def execute(self, state_below, mask=None, cond_input=None):
        if self.cond:
            assert cond_input is not None

        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        dim = tparams[_p(prefix, 'Ux')].shape[1]

        if mask is None:
            mask = tensor.alloc(1., state_below.shape[0], 1)

        # utility function to slice a tensor
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]

        # state_below is the input word embeddings
        # input to the gates, concatenated
        state_below_ = tensor.dot(state_below, tparams[self.__tparams('W')]) + \
            tparams[self.__tparams('b')]
        # input to compute the hidden state proposal
        state_belowx = tensor.dot(state_below, tparams[self.__tparams('Wx')]) + \
            tparams[self.__tparams('bx')]

        if self.cond:
            # input to the gates, concatenated
            state_below_cond = tensor.dot(cond_input, tparams[self.__tparams('W_cond')])
            # input to compute the hidden state proposal
            state_belowx_cond = tensor.dot(cond_input, tparams[self.__tparams('Wx_cond')])

        # step function to be used by scan
        # arguments    | sequences |outputs-info| non-seqs
        def _step_slice(*args):
            m_ = args.pop(0)
            x_ = args.pop(0)
            xx_ = args.pop(0)
            if self.cond:
                x_c = args.pop(0)
                xx_c = args.pop(0)
            h_ = args.pop(0)
            U = args.pop(0)
            Ux = args.pop(0)

            preact = tensor.dot(h_, U)
            preact += x_
            preact += x_c

            # reset and update gates
            r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

            # compute the hidden state proposal
            preactx = tensor.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_
            preactx = preactx + xx_c

            # hidden state proposal
            h = tensor.tanh(preactx)

            # leaky integrate and obtain next hidden state
            h = u * h_ + (1. - u) * h
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h

        # prepare scan arguments
        seqs = [mask, state_below_, state_belowx]
        if self.cond:
            seqs = seqs + [state_below_cond, state_belowx_cond]
        init_states = [tensor.alloc(0., n_samples, dim)]
        _step = _step_slice
        shared_vars = [tparams[self.__tparams('U')],
                       tparams[self.__tparams('Ux')]]

        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=init_states,
                                    non_sequences=shared_vars,
                                    name=_p(self.name,'_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
        rval = [rval]
        return rval





