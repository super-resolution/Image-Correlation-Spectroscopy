import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from tifffile import TiffFile
from ics import moving_window_cross_correlation
from utils import *
import ui

#test_dynamic(data2[:,460:560,500:600])

path = r"F:\Daten\Chunchu"

with TiffFile(path+r"\cytochalasinD-cell1-crop1.tif") as tif:
    data = tif.asarray()
with TiffFile(path+ r"\untreated-cell3_crop.tif") as tif:
    data2 = tif.asarray()

data2-=data2.min()
mask = cell_mask(data2)
data2 = data2*mask
data2[np.where(data2==0)] = np.mean(data2)
plt.imshow(data2[0])
plt.show()
window=10

results = moving_window_cross_correlation(data2, window=window)
ui.plot_quiver(results, data2, window=window)
#substract_moving_average_fft(data[:,0:88,0:88])
#substract_moving_average_fft(data2[:,0:110,0:110])#,270:500,550:780])#, data2[:,280:510,510:740])#

