import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


def build_callback_slicer(data, im, vec, vec_map, fig):
    axcolor = 'lightgoldenrodyellow'
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    sframe = Slider(axamp, 'frame', 0.0, float(data.shape[0]), valinit=0.0, valstep=1.0)
    def callback_data_picker(val):
        i = int(sframe.value)
        im.set_data(data[i])
        vec.set_UVC(-vec_map[i,:,:,0], -vec_map[i,:,:,1])
        fig.canvas.draw_idle()
    sframe.on_changed(callback_data_picker)
    return sframe

def plot_quiver(vec_map, data, window=10):
    fig,axes = plt.subplots(1)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    vec_new = np.zeros((int(vec_map.shape[0]/10)+1,vec_map.shape[1], vec_map.shape[2], vec_map.shape[3]))
    add = 0
    if (vec_map.shape[0]%10>0):
        add=1
    for i in range(int(vec_map.shape[0]/10)+add):
        vec_new[i] = np.sum(vec_map[i*10:i*10+10],axis=0)

    vec_new*=2.5
    x = np.linspace(0, vec_map.shape[2]-1, vec_map.shape[2])*4+window/2
    y = np.linspace(0, vec_map.shape[1]-1, vec_map.shape[1])*4+window/2
    X, Y = np.meshgrid(x, y)
    im =plt.imshow(data[0])
    #vec_map = np.sum(vec_map, axis=0)

    vec =plt.quiver(X, Y,-vec_new[0,:,:,0],-vec_new[0,:,:,1],angles='xy', scale=1/2, units="dots")

    axcolor = 'lightgoldenrodyellow'
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    sframe = Slider(axamp, 'frame', 0.0, data.shape[0]-1, valinit=0.0, valstep=1.0)

    def callback_data_picker(val):
        i = int(sframe.val)
        im.set_data(data[i])
        i = int(i/20)
        vec.set_UVC(-vec_new[i, :, :, 0], -vec_new[i, :, :, 1])
        fig.canvas.draw_idle()

    sframe.on_changed(callback_data_picker)

    plt.show()

