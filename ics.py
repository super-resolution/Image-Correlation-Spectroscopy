import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.fft import fft2,ifft2


def twoD_Gaussian(pos, amplitude, xo, yo, sigma_x, sigma_y, offset):
    """
    2D Gaussian function
    """
    x,y = pos
    xo = float(xo)
    yo = float(yo)
    a = 1/(2*sigma_x**2)
    c = 1/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + c*((y-yo)**2)))
    return g.ravel()

def fit_to_gaussian(data):
    """Least square fit for 2D Gaussian"""
    ind = np.argmax(data)
    x0,y0 = int(ind/data.shape[0]),ind%data.shape[1]
    x = np.linspace(0, data.shape[0]-1, data.shape[0])
    y = np.linspace(0, data.shape[1]-1, data.shape[1])
    x, y = np.meshgrid(x, y)
    initial_guess = (data.max(), x0, y0, 2, 2, 0)
    bounds = np.array([[0, np.inf], [x0-2,x0+2], [y0-2,y0+2], [0,4],[0,4],[0,1]])
    try:
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data.flatten(),bounds=bounds.T, p0=initial_guess)
        # data_fitted = twoD_Gaussian((x, y), *popt)
        #
        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(data, cmap=plt.cm.jet, origin='bottom',
        #           extent=(x.min(), x.max(), y.min(), y.max()))
        # ax.contour(x, y, data_fitted.reshape(data.shape[0], data.shape[1]), 8, colors='w')
        # print(popt[1], popt[2])
        # plt.show()
        return popt


    except(RuntimeError):
        print("nothing found here")
        return None



def rearange(im):
    h,w = int(im.shape[0]/2), int(im.shape[1]/2)
    cop = copy.deepcopy(im)
    im[:h, :w] = cop[h:, w:]
    im[h:, w:] = cop[:h, :w]
    im[h:, :w] = cop[:h, w:]
    im[:h, w:] = cop[h:, :w]
    return im

def apply_hamming_window(image):
    window_h = np.hamming(image.shape[0])
    window_v = np.hamming(image.shape[1])
    image = np.multiply(image.T, window_h).T
    return np.multiply(image, window_v)

def power_spec(temp, source):
    data_fft = fft2(temp)
    data_fft2 = np.conj(fft2(source))
    x = np.multiply(data_fft, data_fft2)
    x/=np.abs(x)
    return ifft2(x).real

def fft_cross_correlation(template, source):
    """
    Compute the spectral power density via fft.
    Applies Hamming window to prevent edge effects
    """
    template = apply_hamming_window(np.pad(template,2))
    source = apply_hamming_window(np.pad(source,2))
    temp = np.zeros_like(source)
    temp[:template.shape[0], :template.shape[1]] = template

    result = power_spec(temp, source)
    result = np.fft.fftshift(result)
    return result


def moving_window_cross_correlation(data, window=10):
    """
    Compute the flow in your data sample for a moving window of size window and a sampling rate of 4.
    Results are plotted and not automativally saved yet
    """
    k_range = int((data.shape[1] - window) / 4)
    l_range = int((data.shape[2] - window) / 4)
    vec_map = np.zeros((50 ,k_range,l_range,2))
    for i in range(50):
        index=i
        print(i)
        i*=2
        #row window

        for k in range(k_range):
            k*=4
            for l in range(l_range):
                l*=4
                # done: samplerate 4 px
                # done: cropp 16x16
                data[i] -= data[i].min()

                data[i+1] -= data[i+1].min()
                norm = np.mean(data[i])
                sample = data[i, k:k+window,l:l+window]
                if np.mean(sample)> 2*norm:
                    sample = data[i, k:k+window,l:l+window]
                    # done: time window dt=1 FTM
                    test = np.zeros((window+4, window+4))

                    for j in range(3):
                        j+=1
                        image = data[i+j, k:k+window,l:l+window]
                        test += fft_cross_correlation(sample, image)
                    params = fit_to_gaussian(test)
                    if params is not None:
                        vec_map[index,int(k/4),int(l/4),0] = params[1]-(window/2+2)
                        vec_map[index,int(k/4),int(l/4),1] = params[2]-(window/2+2)
                    # done: cross correlate
                else: #print("skipped", k,l)
                     pass
    #todo: write an update function in the plot for the flow
    return vec_map
