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
                            find_files,
                            log_file_begins,
                            log_file_ends)
from podocytes.image_processing import (crop_region_of_interest,
                                        denoise_image,
                                        filter_by_size,
                                        find_glomeruli,
                                        gradient_of_image,
                                        marker_controlled_watershed,
                                        markers_from_blob_coords,
                                        ground_truth_image)


def main(args):
    time_start = log_file_begins(args)
    # User input arguments are expected to have 1-based indexing
    # we convert to 0-based indexing for the python program logic.
    channel_glomeruli = args.glomeruli_channel_number - 1
    channel_podocytes = args.podocyte_channel_number - 1

    image_filenames = find_files(input_image_dir, ext)
    cellcounter_filenames = find_files(input_counts_dir, ".xml")
    logging.info(f"Found {len(cellcounter_filenames)} xml count files. ")
    logging.info(f"{cellcounter_filenames}")
    for xml_filename in cellcounter_filenames:
        xml_tree = ET.parse(xml_filename)
        xml_image_name = xml_tree.find('.//Image_Filename').text
        image = open_matching_image(image_filenames, xml_image_name)

        glomeruli_view = image[..., channel_glomeruli]
        podocytes_view = image[..., channel_podocytes]

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
            cropping_margin = 10  # in pixels
            centroid_offset = tuple(glom.bbox[dim] - cropping_margin
                                    for dim in range(podocytes_view.ndim))
            glom_subvolume = crop_region_of_interest(glomeruli_view,
                                                     glom.bbox,
                                                     margin=cropping_margin,
                                                     pad_mode='mean')
            glom_subvolume_labels = crop_region_of_interest(glomeruli_labels,
                                                            glom.bbox,
                                                            margin=cropping_margin,
                                                            pad_mode='zeros')
            gt_subvolume = crop_region_of_interest(gt_image,
                                                   glom.bbox,
                                                   margin=cropping_margin,
                                                   pad_mode='zeros')
            if np.sum(gt_subvolume) == 0:
                continue  # since these counts were not from this glomerulus
            podocytes_view = denoise_image(podocytes_view)
            podo_subvolume = crop_region_of_interest(podocytes_view,
                                                   glom.bbox,
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


def open_matching_image(image_filenames, xml_image_name):
        image_filename = match_filenames(image_filenames, xml_image_name)
        images = pims.Bioformats(image_filename)
        image_series_index = match_image_index(images,
                                              xml_image_name,
                                              os.path.basename(image_filename)
                                              )
        if image_series_index == None:
           logging.info("No matching image series found!")
           return None
        logging.info(f"{images.metadata.ImageID(image_series_index)}")
        logging.info(f"{images.metadata.ImageName(image_series_index)}")
        images.series = image_series_index
        images.bundle_axes = 'zyxc'
        return images[0]


def match_filenames(image_filenames, xml_image_name):
    """Match correct image filename for a given CellCounter xml file."""
    for fname in image_filenames:
        if os.path.basename(fname) in xml_image_name:
            return fname
    return None


def match_image_index(images, xml_image_name, basename):
    """Match image series index number for a given CellCounter xml file.

    Parameters
    ----------
    images : pims image object, where images[0] is the image ndarray.
        Input image plus metadata.
    xml_image_name : string
    basename : string

    Returns
    -------
    image_index : int
    """
    if xml_image_name == basename:
        image_index = 0
        return image_index  # file has only one image series which matches.
    else:
        n_image_series = images.metadata.ImageCount()
        for image_index in range(n_image_series):
            images.series = image_index
            name = images.metadata.ImageName(image_index)
            multi_series_name = basename + " - " + name
            if xml_image_name == name:
                return image_index  # return index of matching image
            elif xml_image_name == multi_series_name:
                return image_index  # return index of matching image
            else:
                return None  # no match found


def assessment():
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


if __name__=='__main__':
    args = configure_parser()  # User input arguments
    main(args)
