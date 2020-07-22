import numpy as np
import scipy
import matplotlib.pyplot as plt
from tifffile import TiffFile

def radial_profile(data):
    sx,sy = data.shape
    X, Y = np.ogrid[0:sx, 0:sy]

    r = np.hypot(X - sx / 2, Y - sy / 2)

    rbin = (20 * r / r.max()).astype(np.int)
    radial_mean = scipy.ndimage.mean(data, labels=rbin, index=np.arange(1, rbin.max() + 1))
    return radial_mean

def apply_hamming_window(image):
    window_h = np.hamming(image.shape[0])
    window_v = np.hamming(image.shape[1])
    image = np.multiply(image, window_h)
    return np.multiply(image.T, window_v).T

def phase_corrlation(im1, im2):
    im1 = np.pad(im1, 10)
    im2 = np.pad(im2, 10)
    im1 = apply_hamming_window(im1)
    im2 = apply_hamming_window(im2)
    #todo: multiply window function
    data_fft = np.fft.fftshift(np.fft.fft2(im1/im1.max()))
    data_fft2 = np.conj(np.fft.fftshift(np.fft.fft2(im2/im2.max())))
    x = np.multiply(data_fft, data_fft2)

    k = np.log(np.abs(x))
    z = radial_profile((x/np.abs(x)).real, )

    result = np.fft.ifft2(np.fft.ifftshift(x)).real
    result = np.fft.fftshift(result)

    return z


def substract_moving_average_fft(data):
    plt.imshow(data[0])
    plt.show()
    for k in range(8):
        radial_all = []
        print(k)

        for j in range(50):
            j = 50*k+j
            radial = []
            for i in range(11):
                radial.append(phase_corrlation(data[j].astype(np.float64), data[j+i].astype(np.float64)))#diffusion dominated region
            radial_all.append(np.array(radial))
        radial_all = np.array(radial_all)
        radial_all = np.mean(radial_all,axis=0)
        x = np.arange(0,15,1)
        #plt.plot(x,np.log(np.log(radial_all[2,0:15])))
        plt.plot(x,np.log(radial_all[10,0:15]))
    plt.show()

#test_dynamic(data2[:,460:560,500:600])

path = r"D:\Daten\Chunchu\Chunchu-KDEL"

with TiffFile(path+r"\cytochalasinD-cell1.tif") as tif:
    data = tif.asarray()
with TiffFile(path+ r"\untreated-cell3.tif") as tif:
    data2 = tif.asarray()

substract_moving_average_fft(data[:,500:650,500:650])
substract_moving_average_fft(data2[:,400:800,400:800])#,270:500,550:780])#, data2[:,280:510,510:740])#

