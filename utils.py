import numpy as np
import time
import functools
import cv2
from skimage.segmentation import watershed
from functools import wraps
import matplotlib.pyplot as plt
import scipy


def coroutine(func):
    """Decorator for priming a coroutine (func)"""
    @wraps(func)
    def primer(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen
    return primer

@coroutine
def data_collector(name):
    data = []
    while True:
        current_data = yield
        if current_data is not None:
            data.append(current_data)
        else:
            break
    data = np.array(data)
    print(name, np.mean(data), np.std(data))
    return data


def to_full_uint8_values(data):
    data -= data.min()
    data /= data.max()
    data *= 255
    return data

def compute_result_statistics(results, structs, window):
    w = int(window / 2)
    on_time = data_collector("on_time")
    off_time = data_collector("off_time")
    on_count = data_collector("on_count")
    off_count = data_collector("off_count")
    n = np.zeros(results.shape[0])
    for j in range(len(structs)):  # todo: generator
        fig, axs = plt.subplots(2)
        test_mask_full = structs[j][w:-w, w:-w]
        test_mask = cv2.resize(test_mask_full, (results.shape[2], results.shape[1]))
        test_mask = cv2.dilate(test_mask, np.ones((2, 2), np.uint8))
        for i in range(results.shape[0]):
            n[i] = np.sum(
                np.abs(results[i, :, :, 0:2] * test_mask[:, :, np.newaxis]))  # /len(np.where(results[i,:,:,0:2]!=0))
        if n.max() > 1:
            n = scipy.ndimage.gaussian_filter(n, sigma=3)
            axs[1].imshow(test_mask_full)
            axs[0].plot(n)
            plt.show()
            off_ind = np.where(n > 0.5)[0]
            on_ind = np.where(n < 0.5)[0]
            n_on = 0
            t = 0
            for i in range(on_ind.shape[0]):
                if on_ind[i] - on_ind[i - 1] > 1:
                    n_on += 1
                    on_time.send(t)
                    t = 0
                else:
                    t += 1
            n_off = 0
            t = 0
            for i in range(off_ind.shape[0]):
                if off_ind[i] - off_ind[i - 1] > 1:
                    n_off += 1
                    off_time.send(t)
                    t = 0
                else:
                    t += 1
            on_count.send(n_on)
            off_count.send(n_off)
    try:
        off_count.send(None)
    except StopIteration:
        print("success")
    try:
        on_count.send(None)
    except StopIteration:
        print("success")
    try:
        on_time.send(None)
    except StopIteration:
        print("success")
    try:
        off_time.send(None)
    except StopIteration:
        print("success")


def extrude_philapodia_structure(data, mask):
    timal_average = np.mean(data, axis=0)
    timal_average = to_full_uint8_values(timal_average)

    #increase mask size to crop out edges to inner cell
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=6)
    skel = cv2.Canny(timal_average.astype(np.uint8), 90, 140)
    skel = np.uint8(skel * mask)
    #close holes in edge image
    skel = cv2.dilate(skel, np.ones((6, 6), np.uint8), iterations=1)
    skel = cv2.erode(skel, np.ones((6, 6), np.uint8), iterations=1)
    #compute connected components with watershed
    dis = cv2.distanceTransform(skel, cv2.DIST_L2, 3)
    markers = np.zeros_like(dis)
    markers[np.where(np.logical_and(dis > 1, dis < 8))] = 1
    markers = cv2.erode(markers, np.ones((2, 2), np.uint8), iterations=1)
    markers = cv2.dilate(markers, np.ones((4, 4), np.uint8), iterations=1)

    # cv2.threshold(dis, 2, 255,0)[1]
    _, markers = cv2.connectedComponents(markers.astype(np.uint8))
    markers += 1
    colormap = watershed(-dis, markers)
    plt.imshow(colormap)
    plt.show()
    #compute masks and return them
    structs = []
    for j in range(colormap.max()):
        j += 2
        indices = np.where(colormap == j)
        if indices[0].shape[0] > 30:
            mask = np.zeros_like(skel)
            mask[indices] = 1
            structs.append(mask)
    return structs

def cell_mask(data, thresh=1.5):
    """
    Mask the average cell body
    """
    average = np.average(data, axis=0)
    mask = np.zeros((data.shape[1],data.shape[2]))
    mask[np.where(average<thresh*np.mean(average))] = 1
    mask = cv2.erode(mask, np.ones((3,3),np.uint8),iterations=3)
    return mask

def remove_masked_gradient(skel, mask):
    indices = np.array(np.where(skel!=0))
    for i,j in indices.T:
        if i>1 and i <skel.shape[0]-1 and j>1 and j <skel.shape[1]-1:
            if mask[i-1,j]==0 or mask[i+1,j]==0 or mask[i,j-1]==0 or mask[i,j+1]==0 or mask[i-1,j-1]==0 or mask[i+1,j+1]==0  or mask[i-1,j+1]==0  or mask[i+1,j-1]==0:
                skel[i,j] = 0
    return skel

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