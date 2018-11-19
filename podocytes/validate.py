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
                            marker_coords,
                            log_file_begins,
                            log_file_ends)
from podocytes.image_processing import (crop_region_of_interest,
                                        denoise_image,
                                        filter_by_size,
                                        find_glomeruli,
                                        find_podocytes,
                                        gradient_of_image,
                                        marker_controlled_watershed,
                                        markers_from_blob_coords,
                                        ground_truth_image)


def main(args):
    # User input arguments are expected to have 1-based indexing
    # we convert to 0-based indexing for the python program logic.
    channel_glomeruli = args.glomeruli_channel_number - 1
    channel_podocytes = args.podocyte_channel_number - 1

    image_filenames = find_files(args.input_directory, args.file_extension)
    cellcounter_filenames = find_files(args.counts_directory, args.xml_extension)
    logging.info(f"Found {len(cellcounter_filenames)} xml count files. ")
    logging.info(f"{cellcounter_filenames}")
    for xml_filename in cellcounter_filenames:
        xml_tree = ET.parse(xml_filename)
        xml_image_name = xml_tree.find('.//Image_Filename').text
        image = open_matching_image(image_filenames, xml_image_name)
        if image is not None:
            glomeruli_view = image[..., channel_glomeruli]
            podocytes_view = image[..., channel_podocytes]

            podocytes_view = denoise_image(podocytes_view)

            ground_truth_df = marker_coords(xml_tree, 2)
            spatial_columns = ['MarkerZ', 'MarkerY', 'MarkerX']
            ground_truth_img = ground_truth_image(ground_truth_df[spatial_columns].values, glomeruli_view.shape)
            glomeruli_labels = find_glomeruli(glomeruli_view)
            output_filename_all_glom_labels = os.path.join(args.output_directory,
                os.path.splitext(xml_image_name)[0] + "_all_glom_labels.tif")
            io.imsave(output_filename_all_glom_labels,
                      glomeruli_labels.astype(np.uint8))
            glom_regions = filter_by_size(glomeruli_labels,
                                          args.minimum_glomerular_diameter,
                                          args.maximum_glomerular_diameter)
            logging.info(f"{len(glom_regions)} glomeruli identified.")
            logging.info(f"(glom_regions) - label of the selected glomerulus")
            for glom in glom_regions:
                cropping_margin = 10  # in pixels
                centroid_offset = tuple(glom.bbox[dim] - cropping_margin
                                        for dim in range(podocytes_view.ndim))
                glom_subvolume, glom_subvolume_labels, gt_subvolume = \
                    crop_all_images(glomeruli_view, glomeruli_labels, ground_truth_img, glom.bbox)

                if np.sum(gt_subvolume) == 0:
                    continue  # since these counts were not from this glomerulus

                podocyte_regions, centroid_offset, wshed = \
                        find_podocytes(podocytes_view, glom,
                                       cropping_margin=cropping_margin)
                ground_truth_bool = gt_subvolume.astype(np.bool)
                result = wshed[ground_truth_bool]

                ground_truth_podocyte_number = len(ground_truth_df)
                found_label_set = set(wshed.ravel())
                found_label_set.remove(0)  # excludes zero label
                podocyte_number_found = len(found_label_set)

                clicks_mask = (gt_subvolume > 0) * 255
                glom_subvolume_mask = (glom_subvolume_labels > 0) * 255
                output_image = np.stack([wshed.astype(np.uint8),
                                         clicks_mask.astype(np.uint8),
                                         (glom_subvolume * 255).astype(np.uint8),
                                         glom_subvolume.astype(np.uint8),
                                         glom_subvolume_mask.astype(np.uint8)],
                                        axis=1
                                        )  # ImageJ expects input in 'zcyx' format
                output_image = np.expand_dims(output_image, 0)   # create empty time axis
                output_fname = xml_image_name
                output_fname = output_fname.replace(os.sep, '-')
                output_fname = output_fname.replace('.', ' ')
                io.imsave(os.path.join(args.output_directory, output_fname+f"_glom{glom.label}.tif"), output_image, imagej=True)


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
                        help='Extension of image file format (.xml)',
                        type=str, default='.xml')
    parser.add_argument('counts_directory', widget='DirChooser',
                        help='Folder containing Fiji CellCounter files.')
    args = parser.parse_args()
    return args


def open_matching_image(image_filenames, xml_image_name):
    """Find image matching CellCounter xml file and return opened image.

    Parameters
    ----------
    image_filenames : list of str
        List of all image filesnames to search for match.
    xml_image_name : str
        Name to match, recorded in CellCounter xml file.

    Returns
    -------
    images[0] : pims image object
    """
    image_filename = match_filenames(image_filenames, xml_image_name)
    if image_filename:
        images = pims.Bioformats(image_filename)
        image_series_index = match_image_index(images,
                                              xml_image_name,
                                              os.path.basename(image_filename)
                                              )
        if image_series_index == None:
           logging.info("No matching image series found.")
           return None
        logging.info(f"{images.metadata.ImageID(image_series_index)}")
        logging.info(f"{images.metadata.ImageName(image_series_index)}")
        images.series = image_series_index
        images.bundle_axes = 'zyxc'
        return images[0]
    else:
        logging.info("No matching image found.")
        return None


def match_filenames(image_filenames, xml_image_name):
    """Match correct image filename for a given CellCounter xml file.

    Parameters
    ----------
    image_filenames : list of str
        List of all image filesnames to search for match.
    xml_image_name : str
        Name to match, recorded in CellCounter xml file.

    Returns
    -------
    filename : str
        Image filename matching CellCounter xml file.
    """
    for filename in image_filenames:
        if os.path.basename(filename) in xml_image_name:
            return filename
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


def crop_all_images(glomeruli_view, glomeruli_labels, ground_truth_image,
                    bounding_box, cropping_margin=10):
    """

    Parameters
    ----------
    glomeruli_view :
    glomeruli_labels :
    ground_truth_image :
    bounding_box :
    cropping_margin : int, optional


    Returns
    -------
    Named tuple
    """
    glom_subvolume = crop_region_of_interest(glomeruli_view,
                                             bounding_box,
                                             margin=cropping_margin,
                                             pad_mode='mean')
    glom_subvolume_labels = crop_region_of_interest(glomeruli_labels,
                                                    bounding_box,
                                                    margin=cropping_margin,
                                                    pad_mode='zeros')
    ground_truth_subvolume = crop_region_of_interest(ground_truth_image,
                                           bounding_box,
                                           margin=cropping_margin,
                                           pad_mode='zeros')
    name = collections.namedtuple("subvolume",
        ["glomeruli_image", "glomeruli_labels", "ground_truth_image"])
    return name(glom_subvolume, glom_subvolume_labels, ground_truth_subvolume)


def more_logging(glom, ground_truth_podocyte_number, podocyte_number_found):
    logging.info(f"ground_truth_podocyte_number: {ground_truth_podocyte_number}")
    logging.info(f"podocyte_number_found: {podocyte_number_found}")
    logging.info(f"difference: {podocyte_number_found - ground_truth_podocyte_number}")
    logging.info(" ")
    logging.info(f"Glom label: {glom.label}")
    logging.info(f"Glom centroid: {glom.centroid}")
    logging.info(f"Glom bbox: {glom.bbox}")
    logging.info(" ")


if __name__=='__main__':
    args = configure_parser()  # User input arguments
    time_start = log_file_begins(args)
    main(args)
    log_file_ends(time_start)
