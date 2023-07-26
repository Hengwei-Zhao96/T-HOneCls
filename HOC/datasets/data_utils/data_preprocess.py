import numpy as np
import torch


def _np_mean_std_normalize(image, mean, std):
    """
    Args:
        image: 3-D array of shape [height, width, channel]
        mean:  a list or tuple or ndarray
        std: a list or tuple or ndarray
    Returns:
    """
    if not isinstance(mean, np.ndarray):
        mean = np.array(mean, np.float32)
    if not isinstance(std, np.ndarray):
        std = np.array(std, np.float32)
    shape = [1] * image.ndim
    shape[-1] = -1
    return (image - mean.reshape(shape)) / std.reshape(shape)


def _th_mean_std_normalize(image, mean, std):
    """ this version faster than torchvision.transforms.functional.normalize
    Args:
        image: 3-D or 4-D array of shape [batch (optional) , height, width, channel]
        mean:  a list or tuple or ndarray
        std: a list or tuple or ndarray
    Returns:
    """
    shape = [1] * image.dim()
    shape[-1] = -1
    mean = torch.tensor(mean, requires_grad=False).reshape(*shape)
    std = torch.tensor(std, requires_grad=False).reshape(*shape)

    return image.sub(mean).div(std)


def mean_std_normalize(image, mean, std):
    """
    Args:
        image: 3-D array of shape [height, width, channel]
        mean:  a list or tuple
        std: a list or tuple
    Returns:
    """
    print("Data normalization..........")
    if isinstance(image, np.ndarray):
        return _np_mean_std_normalize(image, mean, std)
    elif isinstance(image, torch.Tensor):
        return _th_mean_std_normalize(image, mean, std)
    else:
        raise ValueError('The type {} is not support'.format(type(image)))


def divisible_pad(image_list, size_divisor=128):
    """
    Args:
        image_list: a list of images with shape [channel, height, width]
        size_divisor: int
        to_tensor: whether to convert to tensor
    Returns:
        blob: 4-D ndarray of shape [batch, channel, divisible_max_height, divisible_max_height]
    """
    max_shape = np.array([im.shape for im in image_list]).max(axis=0)

    max_shape[1] = int(np.ceil(max_shape[1] / size_divisor) * size_divisor)
    max_shape[2] = int(np.ceil(max_shape[2] / size_divisor) * size_divisor)

    out = np.zeros([len(image_list), max_shape[0], max_shape[1], max_shape[2]], np.float32)

    for i, resized_im in enumerate(image_list):
        out[i, :, 0:resized_im.shape[1], 0:resized_im.shape[2]] = torch.from_numpy(resized_im)

    return out
