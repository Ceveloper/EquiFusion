import torch
import numpy as np

def numpy_img_to_tensor(np_array):
    return torch.tensor(np_array).permute(2, 0, 1)


def get_fig_as_nparray(fig):
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image