import os
import sys
import time
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('wxagg')
from skimage import io
from . import lifio
import matplotlib.pyplot as plt

from gooey.python_bindings.gooey_decorator import Gooey as gooey
from gooey.python_bindings.gooey_parser import GooeyParser

from skimage.util import invert
from skimage.filters import threshold_otsu, threshold_yen, gaussian
from skimage.morphology import ball, watershed, binary_closing, binary_dilation
from skimage.measure import label, regionprops
from skimage.feature import blob_dog

# Fix from http://chriskiehl.com/article/packaging-gooey-with-pyinstaller/
# Commented out because on py3.6 I get the error 'can't have unbuffered TextIO'
# nonbuffered_stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
# sys.stdout = nonbuffered_stdout


__DESCR__ = ('Load, segment, count, and measure glomeruli and podocytes in '
             'fluorescence images.')


@gooey(default_size=(640, 480),
       image_dir=os.path.join(os.path.dirname(__file__), 'app-images'),
       navigation='TABBED')
def main():
    # Get user input
    parser = GooeyParser(prog='Podocyte Profiler', description=__DESCR__)
    parser.add_argument('Input_directory', widget='DirChooser', nargs='+',
                        help='Folder containing files for processing.')
    parser.add_argument('Output_directory', widget='DirChooser', nargs='+',
                        help='Folder to save output analysis files.')
    parser.add_argument('Glomeruli_channel_number', nargs='+',
                        help='Fluorescence channel with glomeruli.',
                        type=int, default=1)
    parser.add_argument('Podocyte_channel_number', nargs='+',
                        help='Fluorescence channel with podocytes.',
                        type=int, default=2)
    parser.add_argument('Minimum_glomerular_diameter', nargs='+',
                        help='Minimum glomerular diameter (microns).',
                        type=float, default=50)
    parser.add_argument('Maximum_glomerular_diameter', nargs='+',
                        help='Maximum glomerular diameter (microns).',
                        type=float, default=200)
    args = parser.parse_args()
    input_directory = ' '.join(args.Input_directory)
    output_directory = ' '.join(args.Output_directory)
    channel_glomeruli = args.Glomeruli_channel_number[0] - 1
    channel_podocytes = args.Podocyte_channel_number[0] - 1

    # Logging
    timestamp = time.strftime('%d-%b-%Y_%H-%M%p', time.localtime())
    print(timestamp)
    log_filename = os.path.join(output_directory, f"log_podo_{timestamp}")
    logging.basicConfig(
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(f"{log_filename}.log"),
            logging.StreamHandler()
        ])
    logging.info("Podocyte automated analysis program")
    logging.info(f"{timestamp}")

    # Initialize
    try:
        lifio.start()
    except:
        logging.exception('Got exception on the main handler - lifio.start()')
        raise
    summary_stats = pd.DataFrame()
    detailed_stats = pd.DataFrame()

    # Get to work
    for root, _, files in os.walk(input_directory):
        for f in files:
            if f.endswith('.lif'):
                filename = os.path.join(root, f)
                logging.info(f"Processing file: {filename}")
                image_series = lifio.series_iterator(filename, desired_order='TZYXC')
                names, _, resolutions, units = lifio.metadata(filename)
                for image_series_number, image in enumerate(image_series):
                    voxel_dimensions = resolutions[image_series_number]
                    results = process_image(image,
                                            voxel_dimensions,
                                            channel_glomeruli,
                                            channel_podocytes,
                                            args.Minimum_glomerular_diameter[0],
                                            args.Maximum_glomerular_diameter[0])
                    results['image_series_number'] = image_series_number
                    results['image_series_name'] = names[image_series_number]
                    results['image_filename'] = filename
                    results['volume_units_xyz'] = str(units[image_series_number][::-1])
                    detailed_stats = detailed_stats.append(results)
                    output_filename_detailed_stats = os.path.join(
                        output_directory,
                        'Podocyte_detailed_stats_'+timestamp+'.csv')
                    detailed_stats.to_csv(output_filename_detailed_stats)
    # Summarize output and write to file
    summary_stats = create_summary_stats(detailed_stats)
    output_filename_summary_stats = os.path.join(
        output_directory,
        'Podocyte_summary_stats_'+timestamp+'.csv')
    summary_stats.to_csv(output_filename_summary_stats)
    lifio.done()


def process_image(input_image, voxel_dimensions,
                  channel_glomeruli, channel_podocytes,
                  glomerulus_diameter_minimum, glomerulus_diameter_maximum):
    """Process single image volume.

    Expect dimension order TZYXC (time, z-plane, row, column, channel)
    Assume that there is only one timepoint in these datasets
    """
    glomeruli_view = input_image[0, :, :, :, channel_glomeruli]  # assume t=0
    podocytes_view = input_image[0, :, :, :, channel_podocytes]  # assume t=0
    # Denoise images with small gaussian blur
    voxel_volume = np.prod(voxel_dimensions)
    sigma = np.divide(voxel_dimensions[-1], voxel_dimensions)  # non-isotropic
    glomeruli_view = gaussian(glomeruli_view, sigma=sigma)
    podocytes_view = gaussian(podocytes_view, sigma=sigma)
    # Find the glomeruli
    threshold_glomeruli = threshold_yen(glomeruli_view)
    glomeruli_regions = find_glomeruli(glomeruli_view,
                                       threshold_glomeruli,
                                       glomerulus_diameter_minimum,
                                       glomerulus_diameter_maximum)
    logging.info(f"{len(glomeruli_regions)} glomeruli identified.")
    df_image = pd.DataFrame()
    glom_index = 0  # since glom.label is not always sequential
    for glom in glomeruli_regions:
        df = pd.DataFrame()
        podocyte_regions, centroid_offset = find_podocytes(podocytes_view, glom)
        for pod in podocyte_regions:
            real_podocyte_centroid = tuple(pod.centroid[dim] +
                                           centroid_offset[dim]
                                           for dim in range(podocytes_view.ndim))
            # Add interesting statistics to the dataframe
            contents = {'glomeruli_index': glom_index,
                        'glomeruli_label_number': glom.label,
                        'glomeruli_voxel_number': glom.filled_area,
                        'glomeruli_volume': (glom.filled_area * voxel_volume),
                        'glomeruli_equiv_diam_pixels': glom.equivalent_diameter,
                        'glomeruli_centroid_x': glom.centroid[2],
                        'glomeruli_centroid_y': glom.centroid[1],
                        'glomeruli_centroid_z': glom.centroid[0],
                        'podocyte_label_number': pod.label,
                        'podocyte_voxel_number': pod.area,
                        'podocyte_volume': (pod.area * voxel_volume),
                        'podocyte_equiv_diam_pixels': pod.equivalent_diameter,
                        'podocyte_centroid_x': real_podocyte_centroid[2],
                        'podocyte_centroid_y': real_podocyte_centroid[1],
                        'podocyte_centroid_z': real_podocyte_centroid[0],
                        'podocyte_mean_intensity': pod.mean_intensity,
                        'podocyte_max_intensity': pod.max_intensity,
                        'podocyte_min_intensity': pod.min_intensity}
            # Add individual podocyte statistics to dataframe
            df = df.append(contents, ignore_index=True)
        # Add summary statistics (average of all podocytes in glomerulus)
        df['glomeruli_index'] = glom_index
        df['number_of_podocytes'] = len(df)
        df['avg_podocyte_voxel_number'] = np.mean(df['podocyte_voxel_number'])
        df['avg_podocyte_volume'] = np.mean(df['podocyte_volume'])
        df['avg_podocyte_equiv_diam_pixels'] = np.mean(df['podocyte_equiv_diam_pixels'])
        df['avg_podocyte_mean_intensity'] = np.mean(df['podocyte_mean_intensity'])
        df['avg_podocyte_max_intensity'] = np.mean(df['podocyte_max_intensity'])
        df['avg_podocyte_min_intensity'] = np.mean(df['podocyte_min_intensity'])
        glom_index += 1
        df_image = df_image.append(df, ignore_index=True)
    return df_image


def find_glomeruli(glomeruli_image, threshold, min_diameter, max_diameter):
    """Identify glomeruli in the image volume and yield those regions."""
    label_image = label(glomeruli_image > threshold)
    for region in regionprops(label_image):
        if ((region.equivalent_diameter >= min_diameter) and
                (region.equivalent_diameter <= max_diameter)):
            yield region


def find_podocytes(podocyte_image, glomeruli_region,
                   min_sigma=1, max_sigma=4, dog_threshold=0.17):
    """Identify podocytes in the image volume."""
    bbox = glomeruli_region.bbox  # bounding box coordinates
    cropping_margin = 10  # pixels
    centroid_offset = tuple(bbox[dim] - cropping_margin
                            for dim in range(podocyte_image.ndim))
    image_roi = crop_region_of_interest(podocyte_image, bbox,
                                        margin=cropping_margin)
    threshold = threshold_otsu(image_roi)
    mask = image_roi > threshold
    blobs = blob_dog(image_roi,
                     min_sigma=min_sigma,
                     max_sigma=max_sigma,
                     threshold=dog_threshold)
    seeds = blob_dog_image(blobs, image_roi.shape)
    wshed = watershed(invert(image_roi), label(seeds), mask=mask)
    regions = regionprops(wshed, intensity_image=image_roi)
    return (regions, centroid_offset)


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


def blob_dog_image(blobs, image_shape):
    """Create boolean image where pixels labelled True
      match coordinates returned from skimage blob_dog() function."""
    blob_map = np.zeros(image_shape).astype(np.bool)
    for blob in blobs:
        blob_map[int(blob[0]), int(blob[1]), int(blob[2])] = True
    return blob_map


def create_summary_stats(dataframe):
    """Return dataframe with average podocyte statistics per glomerulus."""
    summary_columns = ['image_filename',
                       'image_series_name',
                       'image_series_number',
                       'volume_units_xyz',
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
                       'avg_podocyte_mean_intensity',
                       'avg_podocyte_max_intensity',
                       'avg_podocyte_min_intensity']
    summary_dataframe = dataframe[summary_columns].drop_duplicates()
    return summary_dataframe


if __name__ == '__main__':
    main()
