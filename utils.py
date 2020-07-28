import numpy as np

def cell_mask(data, thresh=1.5):
    """
    Mask the average cell body
    """
    average = np.average(data, axis=0)
    mask = np.zeros((data.shape[1],data.shape[2]))
    mask[np.where(average<thresh*np.mean(average))] = 1
    return mask