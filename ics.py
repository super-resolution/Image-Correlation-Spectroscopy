import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.fft import fft2,ifft2
from utils import timeit



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
    """Least square fit for 2D Gaussian
       edit 29.07. cropped feature space to enhance optimization speed
    """
    x = np.linspace(0, 8, 9)
    y = np.linspace(0, 8, 9)
    x_half = int(data.shape[0]/2)
    y_half = int(data.shape[1]/2)
    #crop feature space to enhance speed
    data = data[x_half-4:x_half+5, y_half-4:y_half+5]
    #receive max index of flattened data
    ind = np.argmax(data)
    #compute row and column of max index
    x0,y0 = int(ind/data.shape[0]),ind%data.shape[1]
    #grid to fit on scipy needs this....
    x, y = np.meshgrid(x, y)
    initial_guess = (data.max(), x0, y0, 1.5, 1.5, 0)
    bounds = np.array([[0, np.inf], [x0-2,x0+2], [y0-2,y0+2], [0,4],[0,4],[0,1]])
    try:#if fit fails catch exception
        #fit to function twoD_Gaussian
        #use bounds for lower and upper limits
        #use initial guess as starting point
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data.flatten(),bounds=bounds.T, p0=initial_guess, maxfev=100)
        # data_fitted = twoD_Gaussian((x, y), *popt)
        #
        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(data, cmap=plt.cm.jet, origin='bottom',
        #           extent=(x.min(), x.max(), y.min(), y.max()))
        # ax.contour(x, y, data_fitted.reshape(data.shape[0], data.shape[1]), 8, colors='w')
        # print(popt[1], popt[2])
        # plt.show()
        popt[1] += x_half-4
        popt[2] += y_half-4
        return popt


    except(RuntimeError):
        print("nothing found here")
        return None



def rearange(im):
    """
    rearange k space to set 0,0 to the center of the image
    """
    h,w = int(im.shape[0]/2), int(im.shape[1]/2)
    cop = copy.deepcopy(im)
    im[:h, :w] = cop[h:, w:]
    im[h:, w:] = cop[:h, :w]
    im[h:, :w] = cop[:h, w:]
    im[:h, w:] = cop[h:, :w]
    return im

def apply_hamming_window(image):
    """Cross correlate after applying hamming window to compensate side effects"""
    window_h = np.hamming(image.shape[0])
    window_v = np.hamming(image.shape[1])
    image = np.multiply(image.T, window_h).T
    return np.multiply(image, window_v)

def power_spec(temp, source):
    """
    Compute power spectrum via fft. Much fast than a convolution
    """
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

def moving_window_cross_correlation(data, window=10, full_norm=0):
    """
    Compute the flow in your data sample for a moving window of size window and a sampling rate of 4.
    Results are plotted and not automativally saved yet
    """
    k_range = int((data.shape[1] - window) / 4)
    l_range = int((data.shape[2] - window) / 4)
    vec_map = np.zeros((200 ,k_range,l_range,2))
    for i in range(200):
        # iterate a fraction of the time series for test purposes
        index=i
        print(i)
        i*=2
        #row window

        for k in range(k_range):
            # row step
            k*=4
            for l in range(l_range):
                # column step
                l*=4
                # done: samplerate 4 px
                # done: cropp 10x10
                data[i] -= data[i].min()

                threshold = np.mean(data[i])
                sample = data[i, k:k+window,l:l+window]
                if np.mean(sample)> 2*threshold :
                    sample = data[i, k:k+window,l:l+window]
                    # done: time window dt=1 FTM
                    test = np.zeros((window+4, window+4))

                    for j in range(3):
                        j+=1
                        # accumulate cross correlation over multiple time frames
                        data[i + j] -= data[i + j].min()

                        image = data[i+j, k:k+window,l:l+window]
                        test += fft_cross_correlation(sample, image)
                    params = fit_to_gaussian(test)
                    if params is not None:
                        #write fit parameters to array
                        vec_map[index,int(k/4),int(l/4),0] = params[1]-(window/2+2)
                        vec_map[index,int(k/4),int(l/4),1] = params[2]-(window/2+2)
                    # done: cross correlate
                else: #print("skipped", k,l)
                     pass
    # done: write an update function in the plot for the flow
    # return parameter map
    return vec_map
