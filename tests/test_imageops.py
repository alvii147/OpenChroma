import numpy as np
import pytest

from openchroma.imageops import (
    open_image,
    save_image,
    split_channels,
    combine_channels,
    crop_image,
    sliding_window,
)


def generate_random_image(height, width):
    img = np.around(np.random.rand(height, width, 3) * 255)

    return img


def test_open_image_save_image():
    img = open_image('docs/img/popcat.png')
    save_image(img, 'docs/img/popcat2.png')


def test_split_channels_combine_channels():
    img = generate_random_image(100, 100)
    img_shape = np.shape(img)

    r, g, b = split_channels(img)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            assert r[i][j] == img[i][j][0]

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            assert g[i][j] == img[i][j][1]

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            assert b[i][j] == img[i][j][2]

    img_combined = combine_channels(r, g, b)

    assert np.array_equal(img, img_combined)


sliding_window_parameters = [
    [
        np.array(
            [
                [0, 9, 18],
                [27, 36, 45],
                [54, 63, 72],
                [81, 90, 99],
            ],
            dtype=np.float64,
        ),
        (2, 2),
        np.mean,
        np.float64,
        False,
        np.array(
            [
                [18, 27],
                [45, 54],
                [72, 81],
            ],
            dtype=np.float64,
        ),
    ],
    [
        np.array(
            [
                [0, 9, 18],
                [27, 36, 45],
                [54, 63, 72],
                [81, 90, 99],
            ],
            dtype=np.float64,
        ),
        (2, 2),
        np.mean,
        np.float64,
        True,
        np.array(
            [
                [0, 4.5, 13.5, 18],
                [13.5, 18, 27, 31.5],
                [40.5, 45, 54, 58.5],
                [67.5, 72, 81, 85.5],
                [81, 85.5, 94.5, 99],
            ],
            dtype=np.float64,
        ),
    ],
]


@pytest.mark.parametrize(
    'img, window, op, dtype, edges, output_img',
    sliding_window_parameters,
)
def test_sliding_window(img, window, op, dtype, edges, output_img):
    output_img_computed = sliding_window(
        img,
        window,
        op=op,
        dtype=dtype,
        edges=edges,
    )
    assert np.array_equal(output_img, output_img_computed)


crop_image_parameters = [
    [
        np.array(
            [
                [0, 5, 10, 15, 20],
                [25, 30, 35, 40, 45],
                [50, 55, 60, 65, 70],
                [75, 80, 85, 90, 95],
            ],
            dtype=np.float64,
        ),
        (1, 1),
        (2, 2),
        None,
        np.array(
            [
                [30, 35],
                [55, 60],
            ],
            dtype=np.float64,
        ),
    ],
    [
        np.array(
            [
                [0, 5, 10, 15, 20],
                [25, 30, 35, 40, 45],
                [50, 55, 60, 65, 70],
                [75, 80, 85, 90, 95],
            ],
            dtype=np.float64,
        ),
        (1, 1),
        (3, 2),
        None,
        np.array(
            [
                [30, 35],
                [55, 60],
                [80, 85],
            ],
            dtype=np.float64,
        ),
    ],
    [
        np.array(
            [
                [0, 5, 10, 15, 20],
                [25, 30, 35, 40, 45],
                [50, 55, 60, 65, 70],
                [75, 80, 85, 90, 95],
            ],
            dtype=np.float64,
        ),
        (0, 1),
        None,
        (2, 2),
        np.array(
            [
                [5, 10],
                [30, 35],
            ],
            dtype=np.float64,
        ),
    ],
    [
        np.array(
            [
                [0, 5, 10, 15, 20],
                [25, 30, 35, 40, 45],
                [50, 55, 60, 65, 70],
                [75, 80, 85, 90, 95],
            ],
            dtype=np.float64,
        ),
        (1, 2),
        None,
        (3, 2),
        np.array(
            [
                [35, 40],
                [60, 65],
                [85, 90],
            ],
            dtype=np.float64,
        ),
    ],
]


def test_crop_image_error():
    with pytest.raises(ValueError):
        crop_image(
            generate_random_image(100, 100),
            (1, 2),
            bottom_right=None,
            height_width=None,
        )

    with pytest.raises(ValueError):
        crop_image(
            generate_random_image(100, 100),
            (1, 2),
            bottom_right=(3, 4),
            height_width=(5, 6),
        )


@pytest.mark.parametrize(
    'img, top_left, bottom_right, height_width, cropped_img',
    crop_image_parameters,
)
def test_crop_image(img, top_left, bottom_right, height_width, cropped_img):
    cropped_img_computed = crop_image(
        img,
        top_left,
        bottom_right=bottom_right,
        height_width=height_width,
    )
    assert np.array_equal(cropped_img, cropped_img_computed)


def test_crop_image_sliding_window():
    img_shape = (np.random.randint(20, 1000), np.random.randint(20, 1000))
    window = (
        np.random.randint(2, img_shape[0] // 2),
        np.random.randint(2, img_shape[1] // 2),
    )
    img = generate_random_image(*img_shape)
    slidingMean = sliding_window(
        img,
        window,
        op=np.mean,
        dtype=np.float64,
        edges=False,
    )

    for i in range(img_shape[0] - window[0] + 1):
        for j in range(img_shape[1] - window[1] + 1):
            np.mean(
                crop_image(
                    img,
                    (i, j),
                    height_width=window,
                )
            ) == slidingMean[i][j]
