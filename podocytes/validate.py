import os
import sys
import time
import logging
import collections
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from skimage import io
import matplotlib as mpl
mpl.use('wxagg')
import pims
from gooey.python_bindings.gooey_decorator import Gooey as gooey
from gooey.python_bindings.gooey_parser import GooeyParser

from podocytes.__init__ import __version__
from podocytes.util import (configure_parser_default,
                            parse_args,
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
                                        ground_truth_image)


def main(args):
    """Compare podocyte counts between software and CellCounter markers."""
    image_filenames = find_files(args.input_directory,
                                 args.file_extension)
    cellcounter_filenames = find_files(args.counts_directory,
                                       args.xml_extension)
    logging.info(f"Found {len(cellcounter_filenames)} xml count files. ")
    all_statistics = []
    for xml_filename in cellcounter_filenames:
        xml_tree = ET.parse(xml_filename)
        xml_image_name = xml_tree.find('.//Image_Filename').text
        filename, image = open_matching_image(image_filenames, xml_image_name)
        if image is not None:
            single_image_stats = process_image(args, image, xml_tree)
            single_image_stats['image_filename'] = filename
            single_image_stats['xml_filename'] = xml_filename
            all_statistics.append(single_image_stats)
    try:
        podocyte_comparison_stats = pd.concat(all_statistics,
                                              ignore_index=True, copy=False)
        output_csv_filename = os.path.join(args.output_directory,
                                           'Podocyte_validation_stats.csv')
        podocyte_comparison_stats.to_csv(output_csv_filename)
    except ValueError as err:
        logging.warning("Empty list can't be concatenated.")
        logging.warning(f'{str(type(err))[8:-2]}: {err}')
    else:
        return podocyte_comparison_stats


__DESCR__ = ('Compare podocyte counts between software and CellCounter files.')
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
    args = parse_args(parser)
    return args


def process_image(args, image, xml_tree, crop_margin=10):
    """Compare podocyte counts between Cellcounter xml and matching image.

    Parameters
    ----------
    args : user input arguments
    image : image array
    xml_tree : xml tree of CellCounter marker file content
    crop_margin : int, optional.
        How many pixels for the margin around each glomerulus when cropping.

    Returns
    -------
    single_image_stats : pandas dataframe with comparison of podocyte counts.
    """
    # Ground truth from Cellcounter xml file
    image_shape = image[..., args.podocyte_channel_number].shape
    ground_truth = cellcounter_ground_truth(xml_tree, image_shape)
    podocyte_number_ground_truth = len(ground_truth.dataframe)
    # Find glomeruli in the image ourselves
    glomeruli_labels = find_glomeruli(image[..., args.glomeruli_channel_number])
    glom_regions = filter_by_size(glomeruli_labels,
                                  args.minimum_glomerular_diameter,
                                  args.maximum_glomerular_diameter)
    logging.info(f"{len(glom_regions)} glomeruli identified.")
    # Count the podocytes
    podocytes_view = denoise_image(image[..., args.podocyte_channel_number])
    single_image_stats = []
    for glom in glom_regions:
        cropped = crop_multiple_images(args,
                                       image,
                                       ground_truth.image,
                                       glomeruli_labels,
                                       glom.bbox,
                                       cropping_margin=10)
        # Check ground truth counts came from this particular glomerulus
        if np.sum(cropped.ground_truth_image) > 0:
            podocyte_regions, centroid_offset, watershed = find_podocytes(
                podocytes_view, glom, cropping_margin=crop_margin)
            podocyte_number_counted = count_podocytes_in_label_image(watershed)
            stats = comparison_statistics(glom,
                                          podocyte_number_ground_truth,
                                          podocyte_number_counted)
            single_image_stats.append(stats)
            filename_output_image = save_validation_images(args,
                                                           watershed,
                                                           cropped,
                                                           xml_tree,
                                                           glom)
        else:
            logging.info("CellCounter markers don't match this glomerulus.")
            continue
    try:
        single_image_stats = pd.concat(single_image_stats,
                                       ignore_index=True, copy=False)
    except ValueError as err:
        logging.warning("Empty list can't be concatenated.")
        logging.warning(f'{str(type(err))[8:-2]}: {err}')
    else:
        return single_image_stats


def comparison_statistics(glom_region,
                          podocyte_number_ground_truth,
                          podocyte_number_counted):
    """Create DataFrame with single glomerulus podocyte count comparison.

    Parameters
    ----------
    glom_region : skimage regionprops object representing the glomerulus.
    podocyte_number_ground_truth : number of podocytes in CellCounter xml file.
    podocyte_number_counted : number of podocytes counted by this software.

    Returns
    -------
    stats : pandas dataframe with single glomerulus podocyte count comparison.
    """
    column_names = ['n_podocytes_ground_truth',
                    'n_podocytes_found',
                    'difference_in_podocyte_number',
                    'gleom_label',
                    'glom_centroid_x',
                    'glom_centroid_y',
                    'glom_centroid_z']
    difference = podocyte_number_ground_truth - podocyte_number_counted
    content = [[podocyte_number_ground_truth,
               podocyte_number_counted,
               difference,
               glom_region.label,
               glom_region.centroid[2],
               glom_region.centroid[1],
               glom_region.centroid[0]]]
    stats = pd.DataFrame(content, columns=column_names)
    return stats


def count_podocytes_in_label_image(label_image):
    """Count number of podocytes in label image.

    Parameters
    ----------
    label_image : Label image of podocyte regions in single glomerulus.

    Returns
    -------
    podocyte_number : int
        Number of podocytes in the label image.
    """
    label_set = set(label_image.ravel())
    label_set.remove(0)  # excludes zero label
    podocyte_number = len(label_set)
    return podocyte_number


def save_validation_images(args, podocyte_watershed, cropped,
                           xml_tree, glom):
    """Save multichannel output validation image.

    Parameters
    ----------
    args : user input arguments
    podocyte_watershed : 3D watershed image showing podoyctes.
    cropped : Named tuple containing cropped.ground_truth_image,
        cropped.podoyctes_image, cropped.glomerulus_image,
        and cropped.glomerulus_labels.
    xml_tree : xml tree from CellCounter file

    Returns
    -------
    output_fname : filename where output validation images are saved.
    """
    cellcounter_clicks = (cropped.ground_truth_image > 0) * 255
    glomerulus_mask = (cropped.glomerulus_labels > 0) * 255
    # ImageJ expects input in 'zcyx' format
    output_image = np.stack([podocyte_watershed.astype(np.uint8),
                             cellcounter_clicks.astype(np.uint8),
                             cropped.podoyctes_image.astype(np.uint8),
                             cropped.glomerulus_image.astype(np.uint8),
                             glomerulus_mask.astype(np.uint8)], axis=1)
    output_image = np.expand_dims(output_image, 0)   # create empty time axis
    output_fname = xml_tree.find('.//Image_Filename').text
    output_fname = output_fname.replace(args.file_extension, '')
    output_fname = output_fname.replace(os.sep, '-')
    output_fname = output_fname.replace('.', ' ')
    io.imsave(os.path.join(args.output_directory,
                           output_fname + f"_glomlabel{glom.label}.tif"),
              output_image, imagej=True)
    return output_fname


def cellcounter_ground_truth(xml_tree, image_shape):
    """Find ground truth dataframe and image from CellCounter xml tree.

    Parameters
    ----------
    xml_tree : xml tree from CellCounter file
    image_shape : tuple, shape of image to match

    Returns
    -------
    ground_truth :named tuple with ground_truth.dataframe, ground_truth.image
    """
    ground_truth_dataframe = marker_coords(xml_tree, 2)
    columns = ['MarkerZ', 'MarkerY', 'MarkerX']
    ground_truth_img = ground_truth_image(
        ground_truth_dataframe[columns].values, image_shape)
    ground_truth = collections.namedtuple("ground_truth",
                                          ["dataframe", "image"])
    return ground_truth(ground_truth_dataframe, ground_truth_img)


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
    filename = match_filenames(image_filenames, xml_image_name)
    if filename:
        images = pims.Bioformats(filename)
        image_series_index = match_image_index(images,
                                               xml_image_name,
                                               os.path.basename(filename))
        if image_series_index is None:
            logging.info("No matching image series found.")
            return None
        logging.info(f"{images.metadata.ImageID(image_series_index)}")
        logging.info(f"{images.metadata.ImageName(image_series_index)}")
        images.series = image_series_index
        images.bundle_axes = 'zyxc'
        return filename, images[0]
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


def crop_multiple_images(args,
                         whole_image,
                         whole_ground_truth_image,
                         whole_glomeruli_labels,
                         bounding_box,
                         cropping_margin=10):
    """Crop multiple input images to the same dimensions, return named tuple.

    Parameters
    ----------
    args : User input arguments
    whole_image : grayscale image of whole image
    whole_ground_truth_image :
    whole_glomeruli_labels : label image of glomeruli regions, filtered by size
    bounding_box : tuple
        Bounding box coordinates as tuple.
        Returned from scikit-image regionprops bbox attribute. Format is:
        3D example (min_pln, min_row, min_col, max_pln, max_row, max_col)
        2D example (min_row, min_col, max_row, max_col)
        Pixels belonging to the bounding box are in the half-open interval.
    cropping_margin : int, optional.
        How many pixels to increase the size of the bounding box by.
        If this margin exceeds the input image array bounds,
        then the output image is padded.

    Returns
    -------
    cropped : Named tuple containing cropped.ground_truth_image,
        cropped.podoyctes_image, cropped.glomerulus_image,
        and cropped.glomerulus_labels.
    """
    whole_glomeruli_view = whole_image[..., args.glomeruli_channel_number]
    whole_podocytes_view = whole_image[..., args.podocyte_channel_number]
    ground_truth_image = crop_region_of_interest(whole_ground_truth_image,
                                                 bounding_box,
                                                 margin=cropping_margin,
                                                 pad_mode='zeros')
    podoyctes_image = crop_region_of_interest(whole_podocytes_view,
                                              bounding_box,
                                              margin=cropping_margin,
                                              pad_mode='mean')
    glomeruli_image = crop_region_of_interest(whole_glomeruli_view,
                                              bounding_box,
                                              margin=cropping_margin,
                                              pad_mode='mean')
    glomeruli_labels = crop_region_of_interest(whole_glomeruli_labels,
                                               bounding_box,
                                               margin=cropping_margin,
                                               pad_mode='zeros')
    cropped = collections.namedtuple("cropped", ["ground_truth_image",
                                                 "podoyctes_image",
                                                 "glomerulus_image",
                                                 "glomerulus_labels"])
    cropped_image_tuple = cropped(ground_truth_image,
                                  podoyctes_image,
                                  glomeruli_image,
                                  glomeruli_labels)
    return cropped_image_tuple


if __name__=='__main__':
    args = configure_parser()  # User input arguments
    time_start = log_file_begins(args)
    main(args)
    log_file_ends(time_start)
