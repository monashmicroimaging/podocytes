import os
import sys
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


@gooey(default_size=(660, 640),
       image_dir=os.path.join(os.path.dirname(__file__), 'app-images'),
       navigation='TABBED')
def main():
    # Get user input
    parser = GooeyParser(prog='Podocyte Profiler', description=__DESCR__)
    parser.add_argument('input_directory', widget='DirChooser', nargs='+',
                        help='Folder containing files for processing.')
    parser.add_argument('output_directory', widget='DirChooser', nargs='+',
                        help='Folder to save output analysis files.')
    parser.add_argument('glomeruli_channel_number', nargs='+',
                        help='Fluorescence channel with glomeruli.',
                        type=int, default=1)
    parser.add_argument('podocyte_channel_number', nargs='+',
                        help='Fluorescence channel with podocytes.',
                        type=int, default=2)
    parser.add_argument('minimum_glomerular_diameter', nargs='+',
                        help='Minimum glomerular diameter (microns).',
                        type=float, default=40)
    parser.add_argument('maximum_glomerular_diameter', nargs='+',
                        help='Maximum glomerular diameter (microns).',
                        type=float, default=300)
    parser.add_argument('file_extension', nargs='+',
                        help='Extension of image file format (.tif, etc.)',
                        type=str, default='.lif')
    args = parser.parse_args()
    input_directory = ' '.join(args.input_directory)
    output_directory = ' '.join(args.output_directory)
    channel_glomeruli = args.glomeruli_channel_number[0] - 1
    channel_podocytes = args.podocyte_channel_number[0] - 1
    arg_min_glom_diameter = args.minimum_glomerular_diameter[0]
    arg_max_glom_diameter = args.maximum_glomerular_diameter[0]
    ext = args.file_extension[0]

    # Logging
    time_start = time.time()
    timestamp = time.strftime('%d-%b-%Y_%H-%M%p', time.localtime())
    print(timestamp)
    log_filename = os.path.join(output_directory, f"log_podo_{timestamp}")
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(f"{log_filename}.log"),
            logging.StreamHandler()
        ])
    logging.info("Podocyte automated analysis program")
    logging.info(f"{timestamp}")
    logging.info("========== USER INOUT ARGUMENTS ==========")
    logging.info(f"input_directory: {input_directory}")
    logging.info(f"output_directory: {output_directory}")
    logging.info(f"glomeruli_channel_number: {args.glomeruli_channel_number}")
    logging.info(f"podocyte_channel_number: {args.podocyte_channel_number}")
    logging.info(f"minimum_glomerular_diameter: {arg_min_glom_diameter}")
    logging.info(f"maximum_glomerular_diameter: {arg_max_glom_diameter}")
    logging.info("======= END OF USER INPUT ARGUMENTS =======")

    # Initialize
    summary_stats = pd.DataFrame()
    detailed_stats = pd.DataFrame()

    # Get to work
    for root, _, files in os.walk(input_directory):
        for f in files:
            if f.endswith(ext):
                filename = os.path.join(root, f)
                logging.info(f"Processing file: {filename}")
                images = pims.open(filename)
                images.bundle_axes = 'zyxc'  # must specify every time!
                n_image_series = images.metadata.ImageCount()
                for im_series_num in range(n_image_series):
                    images.series = im_series_num
                    images.bundle_axes = 'zyxc'
                    logging.info(f"{images.metadata.ImageID(im_series_num)}")
                    logging.info(f"{images.metadata.ImageName(im_series_num)}")
                    voxel_dims = [
                        images.metadata.PixelsPhysicalSizeZ(im_series_num),
                        images.metadata.PixelsPhysicalSizeY(im_series_num),
                        images.metadata.PixelsPhysicalSizeX(im_series_num)
                        ] # plane, row, column order
                    min_glom_diameter = arg_min_glom_diameter / voxel_dims[1]
                    max_glom_diameter = arg_max_glom_diameter / voxel_dims[1]
                    results = process_image(images,
                                            voxel_dims,
                                            channel_glomeruli,
                                            channel_podocytes,
                                            min_glom_diameter,
                                            max_glom_diameter)
                    if results is not None:
                        results['image_series_num'] = images.metadata.ImageID(im_series_num)
                        results['image_series_name'] = images.metadata.ImageName(im_series_num)
                        results['image_filename'] = filename
                        #results['volume_units_xyz'] = str(
                        #    units[im_series_num][::-1]
                        #    )
                        detailed_stats = detailed_stats.append(
                            results, ignore_index=True, sort=False)
                        output_filename_detailed_stats = os.path.join(
                            output_directory,
                            'Podocyte_detailed_stats_'+timestamp+'.csv'
                            )
                        detailed_stats.to_csv(output_filename_detailed_stats)
    # Summarize output and write to file
    if detailed_stats is not None:
        summary_stats = create_summary_stats(detailed_stats)
        output_filename_summary_stats = os.path.join(
            output_directory,
            'Podocyte_summary_stats_'+timestamp+'.csv')
        summary_stats.to_csv(output_filename_summary_stats)
        logging.info(f'Saved statistics to file: '
                     f'{output_filename_summary_stats}')
    time_end = time.time()
    time_delta = time_end - time_start
    minutes, seconds = divmod(time_delta, 60)
    hours, minutes = divmod(minutes, 60)
    logging.info(f'Total runtime: '
                 f'{round(hours)} hours, '
                 f'{round(minutes)} minutes, '
                 f'{round(seconds)} seconds.')
    total_gloms_counted = len(summary_stats)
    if total_gloms_counted > 0:
        minutes_per_glom, seconds_per_glom = divmod(
            time_delta / total_gloms_counted, 60)
        logging.info(f'Average time per glomerulus: '
                     f'{round(minutes_per_glom)} minutes, '
                     f'{round(seconds_per_glom)} seconds.')
    logging.info('Program complete.')



def process_image(input_image, voxel_dimensions,
                  channel_glomeruli, channel_podocytes,
                  glomerulus_diameter_minimum, glomerulus_diameter_maximum):
    """Process single image volume.

    Expect dimension order TZYXC (time, z-plane, row, column, channel)
    Assume that there is only one timepoint in these datasets
    """
    glomeruli_view = input_image[0][..., channel_glomeruli]  # assume t=0
    podocytes_view = input_image[0][..., channel_podocytes]  # assume t=0
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
                        'podocyte_centroid_z': real_podocyte_centroid[0]}
            # Add individual podocyte statistics to dataframe
            df = df.append(contents, ignore_index=True, sort=False)
        if df is not None:
            # Add summary statistics (average of all podocytes in glomerulus)
            df['glomeruli_index'] = glom_index
            df['number_of_podocytes'] = len(df)
            df['avg_podocyte_voxel_number'] = np.mean(
                df['podocyte_voxel_number'])
            df['avg_podocyte_volume'] = np.mean(
                df['podocyte_volume'])
            df['avg_podocyte_equiv_diam_pixels'] = np.mean(
                df['podocyte_equiv_diam_pixels'])
            df['podocyte_density'] = len(df) / (glom.filled_area * voxel_volume)
            df_image = df_image.append(df, ignore_index=True, sort=False)
            logging.info(f"{len(df)} podocytes found for this glomerulus.")
        else:
            logging.info("Zero podocytes found in this glomerulus!")
        glom_index += 1
    return df_image


def find_glomeruli(glomeruli_image, threshold, min_diameter, max_diameter):
    """Identify glomeruli in the image volume and yield those regions."""
    label_image = label(glomeruli_image > threshold)
    regions = []
    for region in regionprops(label_image):
        if ((region.equivalent_diameter >= min_diameter) and
                (region.equivalent_diameter <= max_diameter)):
            regions.append(region)
    return regions


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


if __name__ == '__main__':
    main()
