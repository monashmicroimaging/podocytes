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
from skimage.filters import threshold_otsu, threshold_yen, gaussian
from skimage.morphology import ball, watershed, binary_closing, binary_dilation
from skimage.measure import label, regionprops
from skimage.feature import blob_dog

import tifffile._tifffile  # imported to silence pims warning

from scipy import stats

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
            logging.warning(f'Exception raised when trying to open {filename}:')
            logging.warning(f'{type(err)[8:-2]}: {err}')
            continue  # move on to the next file
        for im_series_num in range(images.metadata.ImageCount()):
            logging.info(f"{images.metadata.ImageID(im_series_num)}")
            logging.info(f"{images.metadata.ImageName(im_series_num)}")
            images.series = im_series_num
            images.bundle_axes = 'zyxc'
            single_image_stats = process_image_series(images, filename, args)
            # Add extra info
            single_image_stats['filename'] = filename
            single_image_stats['image_series_num'] = im_series_num
            
            stats_list.append(single_image_stats)
    output_filename_detailed_stats = os.path.join(args.output_directory,
            'Glomeruli_intensity_quality_stats_' + timestamp + '.csv')
    detailed_stats = pd.concat(stats_list, ignore_index=True, copy=False)
    detailed_stats.to_csv(output_filename_detailed_stats)


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
    parser.add_argument('input_directory', widget='DirChooser',
                        help='Folder containing files for processing.')
    parser.add_argument('output_directory', widget='DirChooser',
                        help='Folder to save output analysis files.')
    parser.add_argument('glomeruli_channel_number',
                        help='Fluorescence channel with glomeruli.',
                        type=int, default=1)
    parser.add_argument('podocyte_channel_number',
                        help='Fluorescence channel with podocytes.',
                        type=int, default=2)
    parser.add_argument('minimum_glomerular_diameter',
                        help='Minimum glomerular diameter (microns).',
                        type=float, default=30)
    parser.add_argument('maximum_glomerular_diameter',
                        help='Maximum glomerular diameter (microns).',
                        type=float, default=300)
    parser.add_argument('file_extension',
                        help='Extension of image file format (.tif, etc.)',
                        type=str, default='.lif')
    args = parser.parse_args()
    return args


def crop_region_of_interest(image, bbox, margin=0, pad_mode='mean'):
    """Return cropped region of interest, with border padding.

    Parameters
    ----------
    image : 3D ndarray
        The input image.
    bbox : tuple
        Bounding box coordinates as tuple.
        Returned from scikit-image regionprops bbox attribute. Format is:
        3D example (min_pln, min_row, min_col, max_pln, max_row, max_col)
        2D example (min_row, min_col, max_row, max_col)
        Pixels belonging to the bounding box are in the half-open interval.
    margin : int, optional
        How many pixels to increase the size of the bounding box by.
        If this margin exceeds the input image array bounds,
        then the output image is padded.
    pad_mode : string, optional
        Type of border padding to use. Is either 'mean' (default) or 'zeros'.

    Returns
    -------
    roi_image : 3D ndarray
        The cropped output array.
    """
    ndims = image.ndim
    max_image_size = np.array([np.size(image, dim) for dim in range(ndims)])
    bbox_min_plus_margin = np.array([coord - margin for coord in bbox[:ndims]])
    bbox_max_plus_margin = np.array([coord + margin for coord in bbox[ndims:]])
    image_min_coords = bbox_min_plus_margin.clip(min=0)
    image_max_coords = bbox_max_plus_margin.clip(max=max_image_size)
    max_roi_size = np.array([abs(bbox_max_plus_margin[dim] -
                                 bbox_min_plus_margin[dim])
                             for dim in range(ndims)])
    roi_min_coord = abs(image_min_coords - bbox_min_plus_margin)
    roi_max_coord = max_roi_size - abs(bbox_max_plus_margin - image_max_coords)
    image_slicer = tuple(slice(image_min_coords[dim], image_max_coords[dim], 1)
                         for dim in range(ndims))
    roi_slicer = tuple(slice(roi_min_coord[dim], roi_max_coord[dim], 1)
                       for dim in range(ndims))
    if pad_mode == 'zeros':
        roi_image = np.zeros(max_roi_size)
    elif pad_mode == 'mean':
        roi_image = np.ones(max_roi_size) * np.mean(image[image_slicer])
    else:
        raise ValueError("'pad_mode' keyword argument unrecognized.")
    roi_image[roi_slicer] = image[image_slicer]
    return roi_image


def denoise_image(image):
    """Denoise images with a slight gaussian blur.

    Parameters
    ----------
    image : 3D ndarray
        Original image data from a single fluorescence channel.

    Returns
    -------
    denoised : 3D ndarray
        Image denoised by slight gaussian blur.
    """
    xy_pixel_size = image.metadata['mpp']
    z_pixel_size = image.metadata['mppZ']
    voxel_dimensions = []
    for i in image.metadata['axes']:
        if i == 'x' or i == 'y':
            voxel_dimensions.append(xy_pixel_size)
        elif i == 'z':
            voxel_dimensions.append(z_pixel_size)
    sigma = np.divide(xy_pixel_size, voxel_dimensions)  # non-isotropic
    denoised = gaussian(image, sigma=sigma)
    return denoised


def find_files(input_directory, ext):
    """Recursive search for filenames matching specified extension.

    Parameters
    ----------
    input_directory : str
        Filepath to input directory location.
    ext : str
        File extension (eg: '.tif', '.lif', etc.)

    Returns
    -------
    filelist : list of str
        List of files matching specified extension in the input directory path.
    """
    filelist = []
    for root, _, files in os.walk(input_directory):
        for f in files:
            # ignore hidden files
            if f.endswith(ext) and not f.startswith('.'):
                filename = os.path.join(root, f)
                filelist.append(filename)
    return filelist


def log_file_begins(args, timestamp):
    """Initialize logging and begin writing log file.

    Parameters
    ----------
    args : argparse arguments
        Input arguments from user.
    timestamp :  str
        Local time as string formatted as day-month-year_hour-minute-AM/PM
    Returns
    -------
    log_filename : str
        Filepath to output log text file location.
    """
    log_filename = os.path.join(args.output_directory, f"quality_assessment_{timestamp}")
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(f"{log_filename}.log"),
            logging.StreamHandler()
        ])
    # Log user input arguments
    logging.info("Podocyte automated analysis program")
    logging.info(f"{timestamp}")
    logging.info("========== USER INPUT ARGUMENTS ==========")
    logging.info(f"input_directory: {args.input_directory}")
    logging.info(f"output_directory: {args.output_directory}")
    logging.info(f"file_extension[0]: {args.file_extension}")
    logging.info(f"glomeruli_channel_number: {args.glomeruli_channel_number}")
    logging.info(f"podocyte_channel_number: {args.podocyte_channel_number}")
    logging.info(f"minimum_glomerular_diameter: {args.minimum_glomerular_diameter}")
    logging.info(f"maximum_glomerular_diameter: {args.maximum_glomerular_diameter}")
    logging.info("======= END OF USER INPUT ARGUMENTS =======")
    return log_filename


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
    minutes, seconds = divmod(time_delta, 60)
    hours, minutes = divmod(minutes, 60)
    logging.info(f'Total runtime: '
                 f'{round(hours)} hours, '
                 f'{round(minutes)} minutes, '
                 f'{round(seconds)} seconds.')
    if total_gloms_counted > 0:
        minutes_per_glom, seconds_per_glom = divmod(
            time_delta / total_gloms_counted, 60)
        logging.info(f'Average time per glomerulus: '
                     f'{round(minutes_per_glom)} minutes, '
                     f'{round(seconds_per_glom)} seconds.')
    logging.info('Program complete.')
    return time_delta


def process_image_series(images, filename, args):
    """Process a single image series to assess staining quality.

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
    image_glomeruli = images[0][..., channel_glomeruli]
    #image_podocytes = images[0][..., channel_podocytes]
    intensity_stats = stats.describe(image_glomeruli, axis=None)
    logging.info(intensity_stats)
    data = {'Number_of_voxels_in_image': [intensity_stats.nobs],
            'Minimum_glomeruli_channel_intensity': [intensity_stats.minmax[0]],
            'Maximum_glomeruli_channel_intensity': [intensity_stats.minmax[1]],
            'Mean_glomeruli_channel_intensity': [intensity_stats.mean],
            'Variance_glomeruli_channel_intensity': [intensity_stats.variance],
            'Skewness_glomeruli_channel_intensity': [intensity_stats.skewness],
            'Kurtosis_glomeruli_channel_intensity': [intensity_stats.kurtosis]}
    single_image_stats = pd.DataFrame(data)
    return single_image_stats


if __name__ == '__main__':
    main()
