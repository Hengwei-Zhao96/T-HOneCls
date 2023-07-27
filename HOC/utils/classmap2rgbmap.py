import numpy as np
import PIL.Image as Image


def classmap2rgbmap(classmap: np.ndarray, palette, cls):
    palette = np.asarray(palette)
    (h, w) = classmap.shape
    rgb = np.zeros((h, w, 3))

    if cls == 'mcc':
        for i in range(h):
            for j in range(w):
                rgb[i, j, :] = palette[classmap[i, j], :]
    else:
        for i in range(h):
            for j in range(w):
                rgb[i, j, :] = palette[classmap[i, j] * cls, :]

    r = Image.fromarray(rgb[:, :, 0]).convert('L')
    g = Image.fromarray(rgb[:, :, 1]).convert('L')
    b = Image.fromarray(rgb[:, :, 2]).convert('L')

    rgb = Image.merge("RGB", (r, g, b))

    return rgb
