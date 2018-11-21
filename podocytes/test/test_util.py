import os
import time
import logging

import pytest
import argparse
from gooey.python_bindings.gooey_parser import GooeyParser

from podocytes import __version__
from podocytes.util import find_files, configure_parser_default


class TestUtil(object):
    def test_find_files(self):
        output = find_files('testdata/findfiles/', '.txt').sort()
        expected = ['file_one.txt', 'file_two.txt'].sort()
        assert output == expected

    def test_configure_parser_default(self):
        parser = GooeyParser()
        parser = configure_parser_default(parser)
        dummy_input = ['/test/input/dir', '/test/output/dir',
                       '1', '2', '30.0', '300.0', '.lif']
        args = parser.parse_args(dummy_input)
        output = list(vars(args)).sort()
        expected = ['input_directory',
                    'output_directory',
                    'glomeruli_channel_number',
                    'podocyte_channel_number',
                    'minimum_glomerular_diameter',
                    'maximum_glomerular_diameter',
                    'file_extension'].sort()
        return output == expected
