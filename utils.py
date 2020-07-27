import numpy as np

def cell_mask(data):
    average = np.average(data, axis=0)
    mask = np.zeros((data.shape[1],data.shape[2]))
    mask[np.where(average<1.5*np.mean(average))] = 1
    return mask