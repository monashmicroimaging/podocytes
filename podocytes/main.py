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
        except Exception:
            logging.warning(f"Exception {Exception} raised "
                            f"when trying to open {filename}")
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
    detailed_stats = pd.concat(stats_list, ignore_index=True, copy=False)
    detailed_stats.to_csv(output_filename_detailed_stats)
    summary_stats = summarize_statistics(detailed_stats,
                                         output_filename_summary_stats,
                                         time_start)


__DESCR__ = ('Load, segment, count, and measure glomeruli and podocytes in '
             'fluorescence images.')
@gooey(default_size=(900, 700),
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


def filter_by_size(label_image, min_diameter, max_diameter):
    """Identify objects within a certain size range & return those regions.

    Uses the equivalent_diameter attribute of regionprops to check size.

    Parameters
    ----------
    label_image : 3D ndarray
        Label image
    min_diameter : float
        Minimum expected size (equivalent diameter of labelled voxels)
        Uses the equivalent_diameter attribute of scikit-image regionprops.
    max_diameter : float
        Maximum expected size (equivalent diameter of labelled voxels)
        Uses the equivalent_diameter attribute of scikit-image regionprops.

    Returns
    -------
    regions : list of RegionProperties
        RegionProperties from scikit-image.
    """
    regions = []
    for region in regionprops(label_image):
        if ((region.equivalent_diameter >= min_diameter) and
                (region.equivalent_diameter <= max_diameter)):
            regions.append(region)
    return regions


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


def find_podocytes(podocyte_image, glomeruli_region,
                   min_sigma=1, max_sigma=4, dog_threshold=0.17):
    """Identify podocytes in the image volume.

    Parameters
    ----------
    podocyte_image : 3D ndarray
        Image of podocyte fluorescence.
    glomeruli_region : RegionProperties
        Single glomeruli region, found with scikit-image regionprops.
    min_sigma : float, optional
        Minimum sigma to find podocyte blobs using difference of gaussians.
    max_sigma : float, optional
        Maximum sigma to find podocyte blobs using difference of gaussians.
    dog_threshold : float, optional
        Threshold value for difference of gaussian blob finding.

    Returns
    -------
    regions : List of RegionProperties
        Region properties for podocytes identified.
    centroid_offset : tuple of int
        Coordinate offset of glomeruli subvolume in image.
    wshed : 3D ndarray
        Watershed image showing podoyctes.
    """
    bbox = glomeruli_region.bbox  # bounding box coordinates
    cropping_margin = 10  # pixels
    centroid_offset = tuple(bbox[dim] - cropping_margin
                            for dim in range(podocyte_image.ndim))
    image_roi = crop_region_of_interest(podocyte_image, bbox,
                                        margin=cropping_margin)
    blobs = blob_dog(image_roi,
                     min_sigma=min_sigma,
                     max_sigma=max_sigma,
                     threshold=dog_threshold)
    wshed = marker_controlled_watershed(image_roi, blobs)
    regions = regionprops(wshed, intensity_image=image_roi)
    return (regions, centroid_offset, wshed)


def glom_statistics(df, glom, glom_index, voxel_volume):
    """Add glomerulus information to podocyte statistics for a single glom.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing values for podocytes.
    glom : RegionProperties
         Glomerulus region properties (from scikit-image regionprops).
    glom_index : int
        Integer label for glomerulus.
    voxel_volume : float
        Real space volume of a single image voxel.

    Returns
    -------
    df : DataFrame
        Input pandas dataframe now containing additional columns
        containing data about the outer glomerulus.
    """
    df['number_of_podocytes'] = len(df)
    df['podocyte_density'] = len(df) / (glom.filled_area * voxel_volume)
    df['glomeruli_index'] = glom_index
    df['glomeruli_label_number'] = glom.label
    df['glomeruli_voxel_number'] = glom.filled_area
    df['glomeruli_volume'] = (glom.filled_area * voxel_volume)
    df['glomeruli_equiv_diam_pixels'] = glom.equivalent_diameter
    df['glomeruli_centroid_x'] = glom.centroid[2]
    df['glomeruli_centroid_y'] = glom.centroid[1]
    df['glomeruli_centroid_z'] = glom.centroid[0]
    return df


def gradient_of_image(image):
    """Take the maximum absolute gradient of the image in all directions."""
    grad = np.gradient(image)  # gradients for individual directions
    grad = np.stack(grad, axis=-1)  # from list of arrays to single numpy array
    gradient_image = np.sum(abs(grad), axis=-1)
    return gradient_image


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
    log_filename = os.path.join(args.output_directory, f"log_podo_{timestamp}")
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


def marker_controlled_watershed(grayscale_image, marker_coords):
    """Returns the watershed result given a grayscale image and marker seeds.

    Parameters
    ----------
    grayscale_image : 3D ndarray
        Input image to apply watershed on.
    marker_coords : 3D ndarray
        Array where the first consecutive elements in each row
        are the spatial coordinates of the markers.

    Returns
    -------
    wshed : 3D ndarray
        Label image of watershed results.
    """
    gradient_image = gradient_of_image(grayscale_image)
    seeds = markers_from_blob_coords(marker_coords, grayscale_image.shape)
    wshed = watershed(gradient_image, seeds)
    wshed[wshed == np.max(seeds)] = 0  # set background area to zero
    return wshed


def markers_from_blob_coords(blobs, image_shape):
    """Make watershed markers from scikit-image blob_dog coordinates.

    Parameters
    ----------
    blobs : (n, image.ndim + 1) ndarray
        Input is the output of skimage.feature.blog_dog()
    image_shape : tuple of int
        Shape of image array used to generate input parameter blobs.

    Returns
    -------
    markers : 2D or 3D ndarray
        Boolean array shaped like image_shape,
        with blob coordinate locations represented by True values.
    """
    markers = np.zeros(image_shape, dtype=bool)
    markers[tuple(blobs[:, :-1].T.astype(int))] = True
    # This assumes the first voxel is part of the background
    markers[0] = np.max(markers) + 1  # must have seed for the background area too
    markers = label(markers)
    return markers


def podocyte_avg_statistics(df):
    """Average podocyte statistics per glomerulus.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing podocyte statistics for a single glomerulus

    Returns
    -------
    df : DataFrame
        Pandas dataframe containing additional series with aggregate statistics
        for podocytes per glomerulus, including
        * the average podocyte voxel number
        * the average volume in real space of the podocytes
        * the average equivalent diameter of podocytes (in pixels)
    """
    df['avg_podocyte_voxel_number'] = np.mean(df['podocyte_voxel_number'])
    df['avg_podocyte_volume'] = np.mean(df['podocyte_volume'])
    df['avg_podocyte_equiv_diam_pixels'] = np.mean(df['podocyte_equiv_diam_pixels'])
    return df


def podocyte_statistics(podocyte_regions, centroid_offset, voxel_volume):
    """Calculate statistics for podocytes.

    Parameters
    ----------
    podocyte_regions : List of RegionProperties
        Region properties for podocytes identified.
    centroid_offset : tuple of int
        Coordinate offset of glomeruli subvolume in image.
    voxel_volume : float
        Real space volume of a single image voxel.

    Returns
    -------
    df : DataFrame or None
        Pandas dataframe containing podocytes statistics,
        or None if no regions matching podocyte criteria were identified.
    """
    column_names = ['podocyte_label_number',
                    'podocyte_voxel_number',
                    'podocyte_volume',
                    'podocyte_equiv_diam_pixels',
                    'podocyte_centroid_x',
                    'podocyte_centroid_y',
                    'podocyte_centroid_z']
    contents_list = []
    for pod in podocyte_regions:
        real_podocyte_centroid = tuple(pod.centroid[dim] +
                                       centroid_offset[dim]
                                       for dim in range(len(pod.centroid)))
        # Add interesting statistics to the dataframe
        # Centroid coords are (x, y, z) and NOT (plane, row, column)
        contents = [pod.label,
                    pod.area,
                    pod.area * voxel_volume,
                    pod.equivalent_diameter,
                    real_podocyte_centroid[2],
                    real_podocyte_centroid[1],
                    real_podocyte_centroid[0]]
        # Add individual podocyte statistics to dataframe
        contents_list.append(contents)
    df = pd.DataFrame(contents_list, columns=column_names)
    return df


def preprocess_glomeruli(glomeruli_view):
    """Preprocess glomeruli channel image, return labelled glomeruli image.

    Parameters
    ----------
    glomeruli_view : 3D ndarray
        Image array of glomeruli fluorescence channel.

    Returns
    -------
    label_image : 3D ndarray
        Label image identifying fluorescence regions in glomeruli channel.
    """
    glomeruli_view = denoise_image(glomeruli_view)
    threshold = threshold_yen(glomeruli_view)
    label_image = label(glomeruli_view > threshold)
    return label_image


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
    glomeruli_labels = preprocess_glomeruli(images[0][..., channel_glomeruli])
    glom_regions = filter_by_size(glomeruli_labels,
                                  args.minimum_glomerular_diameter,
                                  args.maximum_glomerular_diameter)
    glom_index = 0  # glom labels will not always be sequential after filtering by size
    logging.info(f"{len(glom_regions)} glomeruli identified.")
    if len(glom_regions) > 0:
        podocytes_view = denoise_image(images[0][..., channel_podocytes])
        for glom in glom_regions:
            podocyte_regions, centroid_offset, wshed = find_podocytes(
                podocytes_view,
                glom)
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
    single_image_stats = pd.concat(df_list, ignore_index=True, copy=False)
    return single_image_stats


def summarize_statistics(detailed_stats, output_filename, time_start):
    """Return dataframe with average podocyte statistics per glomerulus.

    Parameters
    ----------
    detailed_stats : DataFrame
        Pandas dataframe containing detailed podocyte statistics.
    output_filename : str
        Filepath to save summarized statistics to.
    time_start : datetime
        Time Podo program began running.

    Returns
    -------
    summary_stats : DataFrame
        Pandas dataframe containing average podocyte statistics per glomerulus.
    """
    if len(detailed_stats) > 0:
        summary_columns = ['image_filename',
                           'image_series_name',
                           'image_series_num',
                           'glomeruli_index',
                           'glomeruli_label_number',
                           'glomeruli_voxel_number',
                           'glomeruli_volume',
                           'glomeruli_equiv_diam_pixels',
                           'glomeruli_centroid_x',
                           'glomeruli_centroid_y',
                           'glomeruli_centroid_z',
                           'number_of_podocytes',
                           'avg_podocyte_voxel_number',
                           'avg_podocyte_volume',
                           'podocyte_density']
        summary_stats = detailed_stats[summary_columns].drop_duplicates()
        summary_stats.to_csv(output_filename)
        logging.info(f'Saved summary statistics to file: {output_filename}')
        total_gloms_counted = len(summary_stats)
        log_file_ends(time_start, total_gloms_counted)
        return summary_stats
    else:
        total_gloms_counted = 0
        log_file_ends(time_start, total_gloms_counted)
        return None


if __name__ == '__main__':
    main()
