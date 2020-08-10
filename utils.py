import numpy as np
import time
import functools

def cell_mask(data, thresh=1.5):
    """
    Mask the average cell body
    """
    average = np.average(data, axis=0)
    mask = np.zeros((data.shape[1],data.shape[2]))
    mask[np.where(average<thresh*np.mean(average))] = 1
    return mask

def timeit(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()

        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        arg_lst = []
        if args:
            arg_lst.append(', '.join(repr(arg) for arg in args))
        if kwargs:
            pairs = ['%s=%r' % (k, w) for k, w in sorted(kwargs.items())]
            arg_lst.append(', '.join(pairs))
        arg_str = ', '.join(arg_lst)
        print('[%0.8fs] %s ' % (elapsed, name))
        return result
    return clocked