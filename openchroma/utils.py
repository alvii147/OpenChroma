import numpy as np


def is_array_like(arr):
    '''
    Check if object is array-like.

    Parameters
    ----------
    arr : any
        Object to be tested.

    Returns
    -------
    arr_like : bool
        Indicates whether given object is array-like.
    '''

    arr_like = isinstance(arr, (list, tuple, np.ndarray))

    return arr_like


def require_array_like(arr, var_name='Array', exception=TypeError):
    '''
    Raise exception if object is array-like.

    Parameters
    ----------
    arr : any
        Object to be tested.
    var_name : str, optional
        Name of the variable to be tested, used to construct exception message.
    exception : Exception, optional
        Exception to raise if test fails.
    '''

    if not is_array_like(arr):
        message = f'`{var_name}` must be an array-like object'
        raise exception(message)


def is_shape(arr, shape):
    '''
    Check if array is of desired shape.

    Parameters
    ----------
    arr : array-like
        Array to be tested.
    shape : array-like
        Desired shape.

    Returns
    -------
    shapes_equal : bool
        Indicates whether given array is of desired shape.
    '''

    arr_shape = np.shape(arr)
    if np.isscalar(shape):
        shape = (shape,)

    shapes_equal = np.array_equal(arr_shape, shape)

    return shapes_equal


def require_shape(arr, shape, var_name='Array', exception=ValueError):
    '''
    Raise exception if array is not of desired shape.

    Parameters
    ----------
    arr : any
        Array to be tested.
    shape : array-like
        Desired shape.
    var_name : str, optional
        Name of the variable to be tested, used to construct exception message.
    exception : Exception, optional
        Exception to raise if test fails.
    '''

    if not is_shape(arr, shape):
        message = f'`{var_name}` must be of shape {shape}'
        raise exception(message)


def is_dim(arr, dim):
    '''
    Check if array is of desired shape.

    Parameters
    ----------
    arr : array-like
        Array to be tested.
    dim : int
        Desired dimension.

    Returns
    -------
    dim_equal : bool
        Indicates whether given array is of desired dimension.
    '''

    arr_shape = np.shape(arr)
    dim_equal = len(arr_shape) == dim

    return dim_equal


def require_dim(arr, dim, var_name='Array', exception=ValueError):
    '''
    Raise exception if array is not of desired dimension.

    Parameters
    ----------
    arr : any
        Array to be tested.
    dim : int
        Desired dimension.
    var_name : str, optional
        Name of the variable to be tested, used to construct exception message.
    exception : Exception, optional
        Exception to raise if test fails.
    '''

    if not is_dim(arr, dim):
        message = f'`{var_name}` must be {dim}-dimensional'
        raise exception(message)


def is_axis_size(arr, size, axis=0):
    '''
    Check if given axis of array is of desired size.

    Parameters
    ----------
    arr : any
        Array to be tested.
    size : int
        Desired size.
    axis : int, optional
        Axis of array to test for desired size.

    Returns
    -------
    size_equal : bool
        Indicates whether given axis of array is of desired size.
    '''

    arr_shape = np.shape(arr)
    size_equal = arr_shape[axis] == size

    return size_equal


def require_axis_size(
    arr,
    size,
    axis=0,
    var_name='Array',
    exception=ValueError,
):
    '''
    Raise exception if given axis of array is not of desired size.

    Parameters
    ----------
    arr : any
        Array to be tested.
    size : int
        Desired size.
    axis : int, optional
        Axis of array to test for desired size.
    var_name : str, optional
        Name of the variable to be tested, used to construct exception message.
    exception : Exception, optional
        Exception to raise if test fails.
    '''

    if not is_axis_size(arr, size, axis=axis):
        message = f'`{var_name}` must be of size {size} at axis {axis}'
        raise exception(message)
