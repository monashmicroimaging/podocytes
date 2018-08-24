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

# Fix from http://chriskiehl.com/article/packaging-gooey-with-pyinstaller/
# Commented out because on py3.6 I get the error 'can't have unbuffered TextIO'
# nonbuffered_stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
# sys.stdout = nonbuffered_stdout


__DESCR__ = ('Load, segment, count, and measure glomeruli and podocytes in '
             'fluorescence images.')


@gooey(default_size=(700, 700),
       image_dir=os.path.join(os.path.dirname(__file__), 'app-images'),
       navigation='TABBED')
def main():
    # Get user input
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
    input_directory = ' '.join(args.input_directory)
    output_directory = ' '.join(args.output_directory)
    channel_glomeruli = args.glomeruli_channel_number[0] - 1
    channel_podocytes = args.podocyte_channel_number[0] - 1
    min_glom_diameter = args.minimum_glomerular_diameter
    max_glom_diameter = args.maximum_glomerular_diameter
    ext = args.file_extension[0]

    # Initialize
    time_start = time.time()
    timestamp = time.strftime('%d-%b-%Y_%H-%M%p', time.localtime())
    log_file_begins(output_directory, args, timestamp)
    detailed_stats = pd.DataFrame()
    output_filename_detailed_stats = os.path.join(output_directory,
            'Podocyte_detailed_stats_'+timestamp+'.csv')

    # Get to work
    filelist = find_files(input_directory, ext)
    logging.info(f"{len(filelist)} {ext} files found.")
    for filename in filelist:
        logging.info(f"Processing file: {filename}")
        images = pims.open(filename)
        for im_series_num in range(images.metadata.ImageCount()):
            logging.info(f"{images.metadata.ImageID(im_series_num)}")
            logging.info(f"{images.metadata.ImageName(im_series_num)}")
            images.series = im_series_num
            images.bundle_axes = 'zyxc'
            voxel_volume = images[0].metadata['mpp'] * \
                           images[0].metadata['mpp'] * \
                           images[0].metadata['mppZ']
            glomeruli_labels = preprocess_glomeruli(images[0][..., channel_glomeruli])
            glom_regions = filter_by_size(glomeruli_labels, min_glom_diameter, max_glom_diameter)
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
                    if df is None: continue
                    logging.info(f"{len(df)} podocytes found for this glomerulus. " +
                                 f"Centroid (x,y,z): (" +
                                 f"{int(glom.centroid[2])}, " +
                                 f"{int(glom.centroid[1])}, " +
                                 f"{int(glom.centroid[0])})")
                    df = podocyte_avg_statistics(df)
                    df = glom_statistics(df, glom, glom_index, voxel_volume)
                    detailed_stats = detailed_stats.append(df, ignore_index=True, sort=False)
                    #detailed_stats.to_csv(os.path.join(output_directory, 'detailedstats.csv'))
                    glom_index += 1
            # add image details to dataframe
            if detailed_stats is not None:
                detailed_stats['image_series_num'] = images.metadata.ImageID(im_series_num)
                detailed_stats['image_series_name'] = images.metadata.ImageName(im_series_num)
                detailed_stats['image_filename'] = filename
                detailed_stats.to_csv(output_filename_detailed_stats)
    # Summarize output and write to file
    summary_stats = create_summary_stats(detailed_stats)
    output_filename_summary_stats = os.path.join(output_directory,
            'Podocyte_summary_stats_'+timestamp+'.csv')
    summary_stats.to_csv(output_filename_summary_stats)
    logging.info(f'Saved statistics to file: {output_filename_summary_stats}')
    total_gloms_counted = len(summary_stats)
    log_file_ends(time_start, total_gloms_counted)


def blob_dog_image(blobs, image_shape):
    """Create boolean image where pixels labelled True
      match coordinates returned from skimage blob_dog() function."""
    blob_map = np.zeros(image_shape).astype(np.bool)
    for blob in blobs:
        blob_map[int(blob[0]), int(blob[1]), int(blob[2])] = True
    return blob_map


def create_summary_stats(dataframe):
    """Return dataframe with average podocyte statistics per glomerulus."""
    if len(dataframe) == 0:
        return None
    else:
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
        summary_dataframe = dataframe[summary_columns].drop_duplicates()
        return summary_dataframe


def crop_region_of_interest(image, bbox, margin=0, pad_mode='mean'):
    """Return cropped region of interest, with border padding.

    Parameters
    ----------
    image : (N, M) ndarray
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
    roi_image : (N, M) ndarray
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
    """Denoise images with a slight gaussian blur."""
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
    """Identify objects within a certain size range,
    and return those regions as a list.

    Uses the equivalent_diameter attribute of regionprops."""
    regions = []
    for region in regionprops(label_image):
        if ((region.equivalent_diameter >= min_diameter) and
                (region.equivalent_diameter <= max_diameter)):
            regions.append(region)
    return regions


def find_files(input_directory, ext):
    """Recursive search for filenames matching specified extension."""
    filelist = []
    for root, _, files in os.walk(input_directory):
        for f in files:
            if f.endswith(ext):
                filename = os.path.join(root, f)
                filelist.append(filename)
    return filelist


def find_podocytes(podocyte_image, glomeruli_region,
                   min_sigma=1, max_sigma=4, dog_threshold=0.17):
    """Identify podocytes in the image volume."""
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
    """"""
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


def podocyte_avg_statistics(df):
    """Average podocyte statistics per glomerulus."""
    df['avg_podocyte_voxel_number'] = np.mean(df['podocyte_voxel_number'])
    df['avg_podocyte_volume'] = np.mean(df['podocyte_volume'])
    df['avg_podocyte_equiv_diam_pixels'] = np.mean(df['podocyte_equiv_diam_pixels'])
    return df


def podocyte_statistics(podocyte_regions, centroid_offset, voxel_volume):
    """"""
    df = pd.DataFrame()
    for pod in podocyte_regions:
        real_podocyte_centroid = tuple(pod.centroid[dim] +
                                       centroid_offset[dim]
                                       for dim in range(len(pod.centroid)))
        # Add interesting statistics to the dataframe
        contents = {'podocyte_label_number': pod.label,
                    'podocyte_voxel_number': pod.area,
                    'podocyte_volume': (pod.area * voxel_volume),
                    'podocyte_equiv_diam_pixels': pod.equivalent_diameter,
                    'podocyte_centroid_x': real_podocyte_centroid[2],
                    'podocyte_centroid_y': real_podocyte_centroid[1],
                    'podocyte_centroid_z': real_podocyte_centroid[0]}
        # Add individual podocyte statistics to dataframe
        df = df.append(contents, ignore_index=True, sort=False)
    if len(df) > 0:
        return df
    else:
        return None


def gradient_of_image(image):
    """Direction agnostic."""
    grad = np.gradient(image)  # gradients for individual directions
    grad = np.stack(grad, axis=-1)  # from list of arrays to single numpy array
    gradient_image = np.sum(abs(grad), axis=-1)
    return gradient_image


def log_file_begins(output_directory, args, timestamp):
    """Initialize logging and begin writing log file."""
    log_filename = os.path.join(output_directory, f"log_podo_{timestamp}")
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(f"{log_filename}.log"),
            logging.StreamHandler()
        ])
    # Log user input arguments
    input_directory = ' '.join(args.input_directory)
    output_directory = ' '.join(args.output_directory)
    logging.info("Podocyte automated analysis program")
    logging.info(f"{timestamp}")
    logging.info("========== USER INOUT ARGUMENTS ==========")
    logging.info(f"input_directory: {input_directory}")
    logging.info(f"output_directory: {output_directory}")
    logging.info(f"file_extension[0]: {args.file_extension[0]}")
    logging.info(f"glomeruli_channel_number: {args.glomeruli_channel_number}")
    logging.info(f"podocyte_channel_number: {args.podocyte_channel_number}")
    logging.info(f"minimum_glomerular_diameter: {args.minimum_glomerular_diameter[0]}")
    logging.info(f"maximum_glomerular_diameter: {args.maximum_glomerular_diameter[0]}")
    logging.info("======= END OF USER INPUT ARGUMENTS =======")


def log_file_ends(time_start, total_gloms_counted):
    """"""
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


def marker_controlled_watershed(grayscale_image, marker_coords):
    """Returns the watershed result given a grayscale image and marker seeds.

    Parameters
    ----------
    grayscale_image : (M,N) ndarray

    marker_coords : (M,N) ndarray
        Array where the first consecutive elements in each row
        are the spatial coordinates of the markers.

    Returns
    -------
    wshed : (M,N) ndarray
        Label image of watershed results.
    """
    gradient_image = gradient_of_image(grayscale_image)
    seeds = blob_dog_image(marker_coords, grayscale_image.shape)
    seeds[0] = np.max(seeds) + 1  # must have seed for the background area too
    wshed = watershed(gradient_image, label(seeds))
    wshed[wshed == np.max(seeds)] = 0  # set background area to zero
    return wshed


def preprocess_glomeruli(glomeruli_view):
    """Preprocess glomeruli channel image."""
    glomeruli_view = denoise_image(glomeruli_view)
    threshold = threshold_yen(glomeruli_view)
    label_image = label(glomeruli_view > threshold)
    return label_image


if __name__ == '__main__':
    main()
