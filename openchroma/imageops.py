import numpy as np
from PIL import Image

from .utils import requireArrayLike, requireDim, requireAxisSize, requireShape
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


def cropImage(img, top_left, bottom_right=None, height_width=None):
    '''
    Crop image by given coordinates and lengths.

    Parameters
    ----------
    img : array-like
        2D (or higher) image array.

    top_left : array-like
        Coordinates of top left point where the image should be cropped.

    bottom_right : array-like, optional
        Coordinates of the bottom right point where the image should be
        cropped. If this is not provided, ``height_width`` must be given.

    height_width : array-like, optional
        Height & width of the cropped image, packed in a 2-element array. If
        this is not provided, ``bottom_right`` must be given.

    Returns
    -------
    cropped_img : numpy.ndarray
        2D (or higher) cropped image array.
    '''

    # check if inputs are array-like
    requireArrayLike(img, var_name='img')
    # check if top left coordinates are array-like and of shape (2,)
    requireArrayLike(top_left, var_name='top_left')
    requireShape(top_left, (2,))

    top = top_left[0]
    left = top_left[1]

    if bottom_right is None:
        if height_width is None:
            message = 'At least one of '
            message += '`bottom_right` and `height_width` '
            message += 'must be specified'
            raise ValueError(message)

        # check if height & width are array-like
        requireArrayLike(height_width, var_name='height_width')
        # check if height & width are of shape (2,)
        requireShape(height_width, (2,))

        height = height_width[0]
        width = height_width[1]

        bottom = top + height
        right = left + width
    else:
        # check if bottom right coordinates are array-like
        requireArrayLike(bottom_right, var_name='bottom_right')
        # check if bottom left coordinates are of shape (2,)
        requireShape(bottom_right, (2,))

        bottom = bottom_right[0] + 1
        right = bottom_right[1] + 1

    cropped_img = img[top:bottom, left:right]

    return cropped_img


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
    output_img : numpy.ndarray
        Output image array after sliding window operation. If ``edges`` is
        ``True``, its shape is ``(h + n - 1, w + m - 1)``, otherwise its shape
        is ``(h - n + 1, w - n + 1)``.
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
    output_img = np.zeros(output_shape, dtype=dtype)

    # perform operation on image and store in output array
    p = 0
    for i in range(*top_left_range[0]):
        q = 0
        for j in range(*top_left_range[1]):
            window_range = (
                (max(i, 0), min(i + n, h)),
                (max(j, 0), min(j + m, w)),
            )

            output_img[p][q] = op(
                img[
                    window_range[0][0] : window_range[0][1],
                    window_range[1][0] : window_range[1][1],
                ]
            )
            q += 1

        p += 1

    return output_img
