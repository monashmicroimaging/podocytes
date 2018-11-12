import os
import time
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('wxagg')
from skimage import io
import pims
import matplotlib.pyplot as plt

from gooey.python_bindings.gooey_decorator import Gooey as gooey
from gooey.python_bindings.gooey_parser import GooeyParser

from skimage.util import invert
from skimage.filters import threshold_yen, gaussian
from skimage.morphology import ball, watershed, binary_closing, binary_dilation
from skimage.measure import label, regionprops
from skimage.feature import blob_dog

import tifffile._tifffile  # imported to silence pims warning

from podocytes.__init__ import __version__
from podocytes.util import configure_parser_default, log_file_begins, find_files
from podocytes.image_processing import (crop_region_of_interest,
                                        denoise_image,
                                        filter_by_size,
                                        find_glomeruli,
                                        find_podocytes,
                                        gradient_of_image,
                                        marker_controlled_watershed,
                                        markers_from_blob_coords)
from podocytes.statistics import (glom_statistics,
                                  podocyte_statistics,
                                  podocyte_avg_statistics,
                                  summarize_statistics)


def main():
    args = configure_parser()  # User input arguments
    # User input arguments are expected to have 1-based indexing
    # we convert to 0-based indexing for the python program logic.
    channel_glomeruli = args.glomeruli_channel_number - 1
    channel_podocytes = args.podocyte_channel_number - 1

    # Initialize
    time_start = time.time()
    timestamp = time.strftime('%d-%b-%Y_%H-%M%p', time.localtime())
    log_file_begins(args, timestamp)

    # Get to work
    stats_list = []
    filelist = find_files(args.input_directory, args.file_extension)
    logging.info(f"{len(filelist)} {args.file_extension} files found.")
    for filename in filelist:
        logging.info(f"Processing file: {filename}")
        try:
            images = pims.Bioformats(filename)
        except Exception as err:
            logging.warning(f'Exception raised when trying to open {filename}')
            logging.warning(f'{str(type(err))[8:-2]}: {err}')
            continue  # move on to the next file
        for im_series_num in range(images.metadata.ImageCount()):
            logging.info(f"{images.metadata.ImageID(im_series_num)}")
            logging.info(f"{images.metadata.ImageName(im_series_num)}")
            images.series = im_series_num
            images.bundle_axes = 'zyxc'
            single_image_stats = process_image_series(images, filename, args)
            stats_list.append(single_image_stats)
    # Summarize output and write to file
    output_filename_detailed_stats = os.path.join(args.output_directory,
            'Podocyte_detailed_stats_' + timestamp + '.csv')
    output_filename_summary_stats = os.path.join(args.output_directory,
            'Podocyte_summary_stats_' + timestamp + '.csv')
    try:
        detailed_stats = pd.concat(stats_list, ignore_index=True, copy=False)
    except ValueError as err:
        logging.warning(f'No glomeruli identified in this image.')
        logging.warning(f'{str(type(err))[8:-2]}: {err}')
        return None
    else:
        detailed_stats.to_csv(output_filename_detailed_stats)
        summary_stats = summarize_statistics(detailed_stats,
                                             output_filename_summary_stats)
        if len(summary_stats) > 0:
            total_gloms_counted = len(summary_stats)
        else:
            total_gloms_counted = 0
        log_file_ends(time_start, total_gloms_counted)


__DESCR__ = ('Load, segment, count, and measure glomeruli and podocytes in '
             'fluorescence images.')
@gooey(default_size=(800, 700),
       image_dir=os.path.join(os.path.dirname(__file__), 'app-images'),
       navigation='TABBED')
def configure_parser():
    """Configure parser and add user input arguments.

    Returns
    -------
    args : argparse arguments
        Parsed user input arguments.

    """
    parser = GooeyParser(prog='Podocyte Profiler', description=__DESCR__)
    parser = configure_parser_default(parser)
    args = parser.parse_args()
    return args


def log_file_ends(time_start, total_gloms_counted):
    """Append runtime information to log.

    Parameters
    ----------
    time_start : datetime
        Datetime object from program start time.
    total_gloms_counted : int
        The number of glomeruli identified and analyzed.

    Returns
    -------
    time_delta : datetime.timedelta
        How long the program took to run.
    """
    time_end = time.time()
    time_delta = time_end - time_start
    logging.info(f'Total runtime: '
                 f'{round(time_delta)} seconds.')
    if total_gloms_counted > 0:
        seconds_per_glom = time_delta / total_gloms_counted
        logging.info(f'Average time per glomerulus: '
                     f'{round(seconds_per_glom)} seconds.')
    logging.info('Program complete.')
    return time_delta


def process_image_series(images, filename, args):
    """Process a single image series to count the glomeruli and podocytes.

    Parameters
    ----------
    images : pims image object, where images[0] is the image ndarray.
        Input image plus metadata.
    filename : str
        Input image filename.
    args : user input arguments

    Returns
    -------
    single_image_stats : DataFrame

    """
    df_list = []
    # User input arguments are expected to have 1-based indesxing
    # we convert to 0-based indexing for the python program logic.
    channel_glomeruli = args.glomeruli_channel_number - 1
    channel_podocytes = args.podocyte_channel_number - 1
    voxel_volume = images[0].metadata['mpp'] * \
                   images[0].metadata['mpp'] * \
                   images[0].metadata['mppZ']
    logging.info(f"Voxel volume in real space: {voxel_volume}")
    glomeruli_labels = find_glomeruli(images[0][..., channel_glomeruli])
    glom_regions = filter_by_size(glomeruli_labels,
                                  args.minimum_glomerular_diameter,
                                  args.maximum_glomerular_diameter)
    glom_index = 0  # glom labels will not always be sequential after filtering by size
    logging.info(f"{len(glom_regions)} glomeruli identified.")
    if len(glom_regions) > 0:
        podocytes_view = denoise_image(images[0][..., channel_podocytes])
        for glom in glom_regions:
            podocyte_regions, centroid_offset, wshed = \
                    find_podocytes(podocytes_view, glom)
            df = podocyte_statistics(podocyte_regions,
                                     centroid_offset,
                                     voxel_volume)
            logging.info(f"{len(df)} podocytes found for glomerulus " +
                         f"with centroid voxel coord (x,y,z): (" +
                         f"{int(glom.centroid[2])}, " +
                         f"{int(glom.centroid[1])}, " +
                         f"{int(glom.centroid[0])})")
            if len(df) > 0:
                df = podocyte_avg_statistics(df)
                df = glom_statistics(df, glom, glom_index, voxel_volume)
                df['image_series_num'] = images.metadata.ImageID(images.series)
                df['image_series_name'] = images.metadata.ImageName(images.series)
                df['image_filename'] = filename
                glom_index += 1
                df_list.append(df)
    try:
        single_image_stats = pd.concat(df_list, ignore_index=True, copy=False)
    except ValueError as err:
        logging.warning(f'No glomeruli identified.')
        logging.warning(f'{str(type(err))[8:-2]}: {err}')
    else:
        return single_image_stats


if __name__ == '__main__':
    main()
