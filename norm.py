import numpy as np
import numpy.ma as ma

class Normalize(object):
    """
    Similar to matplotlib.colors.Normalize
    """
    def __init__(self, vmin=None, vmax=None):
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, value):
        if self.vmin is None:
            self.vmin = ma.min(value)
            self.vmax = ma.max(value)
        result = ma.array(value).astype(np.float)
        # ma division is slow, take a shortcut
        resdat = result.data
        resdat -= self.vmin
        resdat /= (self.vmax - self.vmin)
        # remask
        result = ma.array(resdat, mask=result.mask, copy=False)
        return result
