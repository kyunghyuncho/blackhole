import numpy



class BHIterator:

    def __init__(fname='grs1915_all_125ms.dat', batch_size=16, shuffle=False):
        self.fname = fname

        self.original_data = numpy.load(fname)
        self.timestamps = [dd[0][0,0] for dd in self.original_data]
        self.signals = [(dd[0][:,1:], dd[1]) for dd in self.original_data]

        self.n_samples = len(self.signals)
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.shuf_idx = numpy.random.permutation(self.n_samples)

        self.n = 0

    def __iter__(self):
        return self

    def reset(self):
        self.shuf_idx = numpy.random.permutation(self.n_samples)
        self.n = 0

    def next(self):
        if self.n > self.n_samples:
            self.reset()
            raise StopIteration

        idx = self.shuf_idx[self.n:self.n+self.batch_size]
        self.n = self.n + self.batch_size

        X = [self.signals[ii][0] for ii in idx]
        Y = [self.signals[ii][1] for ii in idx]

        return X, Y

