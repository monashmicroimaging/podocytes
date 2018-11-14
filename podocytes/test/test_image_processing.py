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
    images = pims.open(filename)
    images.bundle_axes = 'zyxc'
    return images[0]

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
        output = crop_region_of_interest(image, bbox, margin=1, pad_mode='mean')
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
        output = crop_region_of_interest(image, bbox, margin=1, pad_mode='zeros')
        expected = np.array([[0., 0., 0., 0.],
                             [0., 1., 2., 3.],
                             [0., 1., 2., 3.],
                             [0., 1., 2., 3.]])
        assert output.all() == expected.all()

    def test_crop_roi_bad_kwarg(self):
        image = np.random.random((32, 32, 32))
        bbox = (0, 0, 0, 16, 16, 16)
        with pytest.raises(ValueError) as e_info:
            output = crop_region_of_interest(image, bbox, pad_mode='bad_kwarg')


class TestImageProcessing(object):
    pass
    # def test_denoise_image(self):
    #     image = open_test_image()
    #     output = denoise_image(image)
    #     expected =
    #     assert output == expected  # np.allclose()
    #
    # def test_filter_by_size(self):
    #     output =
    #     expected =
    #     assert output == expected
    #
    # def test_find_glomeruli(self):
    #     output =
    #     expected =
    #     assert output == expected
    #
    # def test_find_podocytes(self):
    #     output =
    #     expected =
    #     assert output == expected
    #
    # def test_gradient_of_image(self):
    #     output =
    #     expected =
    #     assert output == expected
    #
    # def test_marker_controlled_watershed(self):
    #     output =
    #     expected =
    #     assert output == expected
    #
    # def test_markers_from_blob_coords(self):
    #     output =
    #     expected =
    #     assert output == expected
