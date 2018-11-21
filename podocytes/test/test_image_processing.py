import os

import pytest
import pims
import numpy as np

from podocytes.image_processing import (crop_region_of_interest,
                                        denoise_image,
                                        filter_by_size,
                                        find_glomeruli,
                                        find_podocytes,
                                        gradient_of_image,
                                        marker_controlled_watershed,
                                        markers_from_blob_coords)

blank_image = np.zeros((128, 128, 128))


def open_test_image():
    fname = 'testdata/51715_glom6.tif'
    filename = os.path.join(os.path.dirname(__file__), fname)
    images = pims.Bioformats(filename)
    images.bundle_axes = 'zyxc'
    return images[0]


class TestFindGlomeruli(object):
    def test_find_glomeruli(self):
        image = open_test_image()
        glomeruli_view = image[..., 0]
        label_image = find_glomeruli(glomeruli_view)
        glomeruli_regions = filter_by_size(label_image, 30.0, 300.0)
        output = len(glomeruli_regions)
        expected = 1
        assert output == expected


class TestCropRegionOfInterest(object):
    def test_crop_region_of_interest(self):
        image = np.random.random((32, 32, 32))
        bbox = (0, 0, 0, 16, 16, 16)
        output = crop_region_of_interest(image, bbox)
        expected = image[:16, :16, :16]
        assert output.all() == expected.all()

    def test_crop_roi_mean_padding(self):
        image = np.array([[1, 2, 3],
                          [1, 2, 3],
                          [1, 2, 3]])
        bbox = (0, 0, 2, 2)
        output = crop_region_of_interest(image, bbox, margin=1,
                                         pad_mode='mean')
        expected = np.array([[2., 2., 2., 2.],
                             [2., 1., 2., 3.],
                             [2., 1., 2., 3.],
                             [2., 1., 2., 3.]])
        assert output.all() == expected.all()

    def test_crop_roi_zero_padding(self):
        image = np.array([[1, 2, 3],
                          [1, 2, 3],
                          [1, 2, 3]])
        bbox = (0, 0, 2, 2)
        output = crop_region_of_interest(image, bbox, margin=1,
                                         pad_mode='zeros')
        expected = np.array([[0., 0., 0., 0.],
                             [0., 1., 2., 3.],
                             [0., 1., 2., 3.],
                             [0., 1., 2., 3.]])
        assert output.all() == expected.all()

    def test_crop_roi_bad_kwarg(self):
        image = np.random.random((32, 32, 32))
        bbox = (0, 0, 0, 16, 16, 16)
        with pytest.raises(ValueError) as e_info:
            output = crop_region_of_interest(image, bbox,
                                             pad_mode='bad_kwarg')
