import os
import sys
import time
import logging
import collections
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib as mpl
mpl.use('wxagg')
import pims
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from gooey.python_bindings.gooey_decorator import Gooey as gooey
from gooey.python_bindings.gooey_parser import GooeyParser

from skimage import io
from skimage.util import invert
from skimage.filters import threshold_otsu, threshold_yen, gaussian
from skimage.morphology import ball, watershed, binary_closing, binary_dilation
from skimage.measure import label, regionprops
from skimage.feature import blob_dog

from podocytes.__init__ import __version__
from podocytes.util import (configure_parser_default,
                            log_file_begins,
                            find_files)
from podocytes.main import (preprocess_glomeruli,
                            filter_by_size,
                            crop_region_of_interest,
                            markers_from_blob_coords,
                            denoise_image,
                            gradient_of_image,
                            marker_controlled_watershed)


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


    # user input
    input_counts_dir = "/Users/genevieb/Desktop/podo_tests/validation/Markers/51745/"
    input_image_dir = "/Users/genevieb/Desktop/podo_tests/validation/Images/51745/"
    output_dir = "/Users/genevieb/Desktop/podo_tests/validation/validation_image_results/51745/"
    ext = '.tif'

    channel_glomeruli = 0
    channel_podocytes = 1
    min_glom_diameter = 30
    max_glom_diameter = 300

    # Initialize logging and begin writing log file.
    timestamp = time.strftime('%d-%b-%Y_%H-%M%p', time.localtime())
    print(timestamp)
    log_filename = os.path.join(output_dir, f"log_validate_{timestamp}")
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(f"{log_filename}.log"),
            logging.StreamHandler()
        ])

    # images
    image_filenames = find_files(input_image_dir, ext)

    # CellCounter xml
    cellcounter_filelist = find_files(input_counts_dir, ".xml")
    logging.info(f"Found {len(cellcounter_filelist)} xml count files. ")
    logging.info(f"{cellcounter_filelist}")


    for xml_filename in cellcounter_filelist:
        xml_tree = ET.parse(xml_filename)
        xml_image_name = xml_tree.find('.//Image_Filename').text
        image_filename = match_filenames(image_filenames, xml_image_name)
        images = pims.Bioformats(image_filename)
        logging.info(f"Processing file: {image_filename}")
        images.bundle_axes = 'zyxc'

        #image_series_index = match_names(images,
        #                                 xml_image_name,
        #                                 os.path.basename(image_filename)
        #                                 )
        #if image_series_index == None:
        #    logging.info("No matching image series found!")
        #    continue
        #logging.info(f"{images.metadata.ImageID(image_series_index)}")
        #logging.info(f"{images.metadata.ImageName(image_series_index)}")
        #images.series = image_series_index
        #images.bundle_axes = 'zyxc'

        glomeruli_view = images[0][..., channel_glomeruli]
        podocytes_view = images[0][..., channel_podocytes]

        ground_truth_df = marker_coords(xml_tree, 2)
        spatial_columns = ['MarkerZ', 'MarkerY', 'MarkerX']
        gt_image = ground_truth_image(ground_truth_df[spatial_columns].values,
                                      glomeruli_view.shape)

        glomeruli_labels = preprocess_glomeruli(glomeruli_view)
        output_filename_all_glom_labels = os.path.join(output_dir,
            os.path.splitext(xml_image_name)[0] + "_all_glom_labels.tif")
        io.imsave(output_filename_all_glom_labels,
                  glomeruli_labels.astype(np.uint8))
        glom_regions = filter_by_size(glomeruli_labels,
                                      min_glom_diameter,
                                      max_glom_diameter)
        logging.info(f"{len(glom_regions)} glomeruli identified.")
        logging.info(f"(glom_regions) - label of the selected glomerulus")
        for glom in glom_regions:
            bbox = glom.bbox
            cropping_margin = 10  # pixels
            centroid_offset = tuple(bbox[dim] - cropping_margin
                                    for dim in range(podocytes_view.ndim))
            glom_subvolume = crop_region_of_interest(glomeruli_view,
                                                     bbox,
                                                     margin=cropping_margin,
                                                     pad_mode='mean')
            glom_subvolume_labels = crop_region_of_interest(glomeruli_labels,
                                                            bbox,
                                                            margin=cropping_margin,
                                                            pad_mode='zeros')
            gt_subvolume = crop_region_of_interest(gt_image,
                                                   bbox,
                                                   margin=cropping_margin,
                                                   pad_mode='zeros')
            if np.sum(gt_subvolume) == 0:
                continue  # since these counts were not from this glomerulus
            podocytes_view = denoise_image(podocytes_view)
            podo_subvolume = crop_region_of_interest(podocytes_view,
                                                   bbox,
                                                   margin=cropping_margin,
                                                   pad_mode='mean')
            threshold = threshold_otsu(podo_subvolume)
            mask = podo_subvolume > threshold
            blobs = blob_dog(podo_subvolume,
                             min_sigma=1,
                             max_sigma=4,
                             threshold=0.17)
            wshed = marker_controlled_watershed(podo_subvolume, blobs)
            ground_truth_bool = gt_subvolume.astype(np.bool)
            result = wshed[ground_truth_bool]

            found_label_set = set(wshed.ravel())
            found_label_set.remove(0)  # excludes zero label
            ground_truth_label_set = set(gt_image.ravel())
            ground_truth_label_set.remove(0)  # excludes zero label
            fancy_indexed_set = set(result)

            ground_truth_podocyte_number = len(ground_truth_df)
            podocyte_number_found = len(found_label_set)
            logging.info(f"ground_truth_podocyte_number: {ground_truth_podocyte_number}")
            logging.info(f"podocyte_number_found: {podocyte_number_found}")
            logging.info(f"difference: {podocyte_number_found - ground_truth_podocyte_number}")
            logging.info(" ")
            logging.info(f"Glom label: {glom.label}")
            logging.info(f"Glom centroid: {glom.centroid}")
            logging.info(f"Glom bbox: {glom.bbox}")
            logging.info(" ")
            click_but_no_label = list(ground_truth_label_set - fancy_indexed_set)
            labels_with_zero_clicks = found_label_set - fancy_indexed_set
            try:
                labels_with_zero_clicks.remove(0)
            except KeyError:
                pass
            finally:
                labels_with_zero_clicks = list(labels_with_zero_clicks)

            labels_with_one_click = [i for i, count in collections.Counter(result).items() if count == 1]
            labels_with_multiple_clicks = [i for i, count in collections.Counter(result).items() if count > 1]

            n_click_but_no_label = len(click_but_no_label)  # missed podocytes, should have found these
            n_labels_with_zero_clicks = len(labels_with_zero_clicks)  # aren't supposed to exist
            n_labels_with_one_click = len(labels_with_one_click)  # correctly identified
            n_labels_with_multiple_clicks = len(labels_with_multiple_clicks)  # labels should be split up
            logging.info(f"{n_click_but_no_label} - n_click_but_no_label, missed podocytes")
            logging.info(f"{n_labels_with_zero_clicks} - n_labels_with_zero_clicks, aren't supposed to exist")
            logging.info(f"{n_labels_with_one_click} - n_labels_with_one_click, correctly identified")
            logging.info(f"{n_labels_with_multiple_clicks} - n_labels_with_multiple_clicks, labels should be split up")

            logging.info(" ")
            logging.info("click_but_no_label: (gt label numbering)")
            logging.info(f"{click_but_no_label}")
            logging.info("labels_with_zero_clicks:")
            logging.info(f"{labels_with_zero_clicks}")
            logging.info("labels_with_one_click:")
            logging.info(f"{labels_with_one_click}")
            logging.info("labels_with_multiple_clicks:")
            logging.info(f"{labels_with_multiple_clicks}")
            logging.info(" ")
            logging.info(f"result = {result}")
            logging.info(" ")

            clicks_mask = (gt_subvolume > 0) * 255
            glom_subvolume_mask = (glom_subvolume_labels > 0) * 255
            output_image = np.stack([wshed.astype(np.uint8),
                                     clicks_mask.astype(np.uint8),
                                     (podo_subvolume * 255).astype(np.uint8),
                                     glom_subvolume.astype(np.uint8),
                                     glom_subvolume_mask.astype(np.uint8)],
                                    axis=1
                                    )  # ImageJ expects input in 'zcyx' format
            output_image = np.expand_dims(output_image, 0)   # create empty time axis
            output_fname = xml_image_name
            output_fname = output_fname.replace(os.sep, '-')
            output_fname = output_fname.replace('.', ' ')
            io.imsave(os.path.join(output_dir, output_fname+f"_glom{glom.label}.tif"), output_image, imagej=True)

    log_file_ends(time_start)


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
    parser.add_argument('xml_extension',
                        help='Extension of CellCounter file format (.xml)',
                        type=str, default='.xml')
    parser.add_argument('counts_directory', widget='DirChooser',
                        help='Folder containing Fiji CellCounter files.')
    args = parser.parse_args()
    return args


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
    logging.info(f"xml_extension: {args.xml_extension}")
    logging.info(f"counts_directory: {args.counts_directory}")
    logging.info("======= END OF USER INPUT ARGUMENTS =======")
    return log_filename


def log_file_ends(time_start):
    """Append runtime information to log.

    Parameters
    ----------
    time_start : datetime
        Datetime object from program start time.

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
    logging.info('Program complete.')
    return time_delta


def match_filenames(image_filenames, xml_image_name):
    for fname in image_filenames:
        if os.path.basename(fname) in xml_image_name:
            return fname
    return None


def match_names(images, xml_image_name, basename):
    n_image_series = images.metadata.ImageCount()
    for i in range(n_image_series):
        images.series = i
        name = images.metadata.ImageName(i)
        multi_series_name = basename + " - " + name
        if xml_image_name == name:
            return i  # return index of matching image
        elif xml_image_name == multi_series_name:
            return i  # return index of matching image
        else:
            return None  # no match found


def ground_truth_image(ground_truth_coords, image_shape):
    """Create label image where pixels labelled with int > 0 match
       coordinates from skimage blob_dog/blob_log/... function."""
    image = np.zeros(image_shape).astype(np.int32)
    for i, gt_coord in enumerate(ground_truth_coords):
        coord = [slice(int(gt_coord[dim]), int(gt_coord[dim]) + 1, 1)
                 for dim in range(image.ndim)]
        image[coord] = i + 1  # only background pixels labelled zero.
    return image


if __name__=='__main__':
    main()
