import os

import pims
import pandas as pd
from gooey.python_bindings.gooey_parser import GooeyParser

from podocytes.main import process_image_series
from podocytes.util import configure_parser_default


class TestMain(object):
    def test_process_image_series(self):
        # Read test image
        fname = 'testdata/51715_glom6.tif'
        filename = os.path.join(os.path.dirname(__file__), fname)
        images = pims.Bioformats(filename)
        images.bundle_axes = 'zyxc'
        # Setup user input args
        parser = GooeyParser()
        parser = configure_parser_default(parser)
        dummy_input = ['/test/input/dir', '/test/output/dir',
                       '1', '2', '30.0', '300.0', '.lif']
        args = parser.parse_args(dummy_input)
        # Count podocytes
        single_image_stats = process_image_series(images, filename, args)
        found_number_of_podocytes = len(single_image_stats)
        expected_number_of_podocytes = 48
        assert found_number_of_podocytes == expected_number_of_podocytes
