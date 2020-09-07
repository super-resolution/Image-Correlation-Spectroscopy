import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from tifffile import TiffFile
from ics import moving_window_cross_correlation
from utils import *
import ui


path = r"F:\Daten\Chunchu"

with TiffFile(path+r"\treated_cell5_crop.tif") as tif:
    o_data2 = tif.asarray()
# with TiffFile(path+ r"\untreated-cell5_crop.tif") as tif:
#     o_data2 = tif.asarray()

o_data2 -= o_data2.min()
mask1 = cell_mask(o_data2)

mask2 = np.ones_like(mask1)-mask1
data2 = o_data2*mask2
data2[np.where(data2==0)] = np.mean(data2)
data2 /= data2.max()
data2 *= 255
#structs = extrude_philapodia_structure(data2, mask)

window = 10
results = moving_window_cross_correlation(data2, window=window)
n = np.zeros(results.shape[0])
for i in range(results.shape[0]):
    n[i] = np.sum(np.abs(results[i, :, :, 0:2])/np.where(results[i,:,:,0]!=0)[0].shape[0])
n = scipy.ndimage.gaussian_filter(n, sigma=3)
plt.plot(n, label="core")

data2 = o_data2*mask1


# data2[np.where(data2==0)] = np.mean(data2)
# data2 /= data2.max()
# data2 *= 255
# #structs = extrude_philapodia_structure(data2, mask1)
# mask1 = cv2.erode(mask1, np.ones((3, 3), np.uint8), iterations=3)
# data2 *= mask1
# results = moving_window_cross_correlation(data2, window=window, full_norm=np.average(o_data2))
# #compute_result_statistics(results, structs, window)
# n = np.zeros(results.shape[0])
# for i in range(results.shape[0]):
#     n[i] = np.sum(np.abs(results[i, :, :, 0:2])/np.where(results[i,:,:,0]!=0)[0].shape[0])
# n = scipy.ndimage.gaussian_filter(n, sigma=3)
# plt.plot(n, label="philapodia")
plt.legend(loc="upper left")
plt.xlabel("time")
plt.ylabel("dynamics a.u.")
ui.plot_quiver(results, data2, window=window)
#substract_moving_average_fft(data[:,0:88,0:88])
#substract_moving_average_fft(data2[:,0:110,0:110])#,270:500,550:780])#, data2[:,280:510,510:740])#

