import numpy as np
from PIL import Image

from .utils import requireArrayLike, requireDim, requireAxisSize
from .constants import RGB_SHAPE


def openImage(path):
    '''
    Open image from given path.

    Parameters
    ----------
    path : str, ``pathlib.Path`` object or file object
        Path to image file.

    Returns
    -------
    img : numpy.ndarray
        3D image array in RGB space.
    '''

    # open image file
    im = Image.open(path)
    # convert image into array
    img = np.array(im.convert('RGB'), dtype=np.float64)

    return img


def saveImage(img, path):
    '''
    Save image at given path.

    Parameters
    ----------
    img : array-like
        3D image array in RGB space.

    path : str, ``pathlib.Path`` object or file object
        Path to file.
    '''

    # check if input is array-like
    requireArrayLike(img, var_name='img')
    # check if last axis is 3-dimensional
    requireAxisSize(img, RGB_SHAPE[-1], axis=-1, var_name='img')

    # create image from array
    im = Image.fromarray(np.array(img, dtype=np.uint8), 'RGB')
    # save image
    im.save(path)


def splitChannels(img):
    '''
    Split image array into RGB channel arrays.

    Parameters
    ----------
    img : array-like
        3D image array in RGB space.

    Returns
    -------
    r : numpy.ndarray
        2D red channel array.

    g : numpy.ndarray
        2D green channel array.

    b : numpy.ndarray
        2D blue channel array.
    '''

    # check if input is array-like
    requireArrayLike(img, var_name='img')
    # check if last axis is 3-dimensional
    requireAxisSize(img, RGB_SHAPE[-1], axis=-1, var_name='img')

    # unpack image array into channel arrays
    r, g, b = np.transpose(img)
    r, g, b = (
        np.transpose(r),
        np.transpose(g),
        np.transpose(b),
    )

    return r, g, b


def combineChannels(r, g, b):
    '''
    Combine RGB channel arrays into image array.

    Parameters
    ----------
    r : array-like
        2D red channel array.

    g : array-like
        2D green channel array.

    b : array-like
        2D blue channel array.

    Returns
    -------
    img : numpy.ndarray
        3D image array in RGB space.
    '''

    # check if inputs are array-like
    requireArrayLike(r, var_name='r')
    requireArrayLike(g, var_name='g')
    requireArrayLike(b, var_name='b')
    # check if inputs are 2-dimensional
    requireDim(r, 2)
    requireDim(g, 2)
    requireDim(b, 2)

    # pack channel arrays into image array
    img = np.stack((r, g, b), axis=-1)

    return img


def slidingWindowOperation(img, window, op=np.mean, dtype=object, edges=False):
    '''
    Perform operation on sliding window over image.

    Parameters
    ----------
    img : array-like
        3D image array in RGB space.

    window : array-like
        2-element array indicating shape of window.

    op : callable function, optional
        Operation to perform on each window.

    dtype : type
        Data type of output array

    edges : bool
        Indicates whether or not to cover edges of image using smaller window.

    Returns
    -------
    output : numpy.ndarray
        Output array after sliding window operation. If ``edges`` is ``True``,
        its shape is ``(h + n - 1, w + m - 1)``, otherwise its shape is
        ``(h - n + 1, w - n + 1)``.
    '''

    h, w = np.shape(img)
    n, m = window

    # set up iteration ranges
    if edges:
        top_left_range = (
            (-n + 1, h),
            (-m + 1, w),
        )
    else:
        top_left_range = (
            (0, h - n + 1),
            (0, w - m + 1),
        )

    # set up output array shape
    output_shape = (
        top_left_range[0][1] - top_left_range[0][0],
        top_left_range[1][1] - top_left_range[1][0],
    )
    # create output array of zeros
    output = np.zeros(output_shape, dtype=dtype)

    # perform operation on image and store in output array
    oi = 0
    for i in range(*top_left_range[0]):
        oj = 0
        for j in range(*top_left_range[1]):
            window_range = (
                (max(i, 0), min(i + n, h)),
                (max(j, 0), min(j + m, w)),
            )

            output[oi][oj] = op(
                img[
                    window_range[0][0] : window_range[0][1],
                    window_range[1][0] : window_range[1][1],
                ]
            )
            oj += 1

        oi += 1

    return output
