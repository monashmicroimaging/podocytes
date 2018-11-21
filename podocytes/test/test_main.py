import os
import argparse

import pims
import pandas as pd

from podocytes.main import process_image_series


class TestMain(object):
    def test_process_image_series(self):
        fname = 'testdata/51715_glom6.tif'
        filename = os.path.join(os.path.dirname(__file__), fname)
        images = pims.Bioformats(filename)
        images.bundle_axes = 'zyxc'
        args = argparse.Namespace(input_directory='/test/input/dir',
                                  output_directory='/test/output/dir',
                                  glomeruli_channel_number=0,
                                  podocyte_channel_number=1,
                                  minimum_glomerular_diameter=30.0,
                                  maximum_glomerular_diameter=300.0,
                                  file_extension='.tif')
        single_image_stats = process_image_series(images, filename, args)
        found_number_of_podocytes = len(single_image_stats)
        expected_number_of_podocytes = 48
        assert found_number_of_podocytes == expected_number_of_podocytes
