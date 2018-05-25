import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use('wxagg')
from skimage import io
import pims
import matplotlib.pyplot as plt

from gooey.python_bindings.gooey_decorator import Gooey as gooey
from gooey.python_bindings.gooey_parser import GooeyParser

# Fix from http://chriskiehl.com/article/packaging-gooey-with-pyinstaller/
# Commented out because on py3.6 I get the error "can't have unbuffered TextIO"
# nonbuffered_stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
# sys.stdout = nonbuffered_stdout


__DESCR__ = ('Load, segment, count, and measure glomeruli and podocytes in '
             'fluorescence images.')


@gooey(default_size=(640, 480),
       image_dir=os.path.join(os.path.dirname(__file__), 'app-images'),
       navigation='TABBED')
def main():
    parser = GooeyParser(prog='Podocyte Profiler', description=__DESCR__)
    parser.add_argument('input_file', help='Files to process',
                        widget='FileChooser', nargs='+')
    parser.add_argument('output_folder', help='Output location',
                        widget='DirChooser')
    args = parser.parse_args()
    filename = ' '.join(args.input_file)
    images = pims.open(filename)
    images.bundle_axes = 'zyxc'  # must specify which dimensionns for frames
    metadata = images.metadata
    n_image_series = metadata.ImageCount()
    print('hooray!')



if __name__ == '__main__':
    main()
