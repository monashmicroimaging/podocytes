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

from skimage import io
from skimage.util import invert
from skimage.filters import threshold_otsu, threshold_yen, gaussian
from skimage.morphology import ball, watershed, binary_closing, binary_dilation
from skimage.measure import label, regionprops
from skimage.feature import blob_dog

from main import preprocess_glomeruli, filter_by_size, \
                 crop_region_of_interest, blob_dog_image, denoise_image, \
                 gradient_of_image

def main():
    # user input
    input_dir = "/Users/genevieb/Desktop/test_podo/little_input/"
    output_dir = "/Users/genevieb/Desktop/test_podo/little_output/"
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

    # CellCounter xml
    xml_filename = os.path.join(input_dir, "CellCounter_51559.xml")
    xml_tree = ET.parse(xml_filename)
    xml_image_name = xml_tree.find('.//Image_Filename').text
    # images
    image_filename = os.path.join(input_dir, "51559.lif")
    images = pims.open(image_filename)
    image_series_index = match_names(images,
                                     xml_image_name,
                                     os.path.basename(image_filename)
                                     )
    if image_series_index == None:
        logging.info("No matching image series found!")
        return
    logging.info(f"{images.metadata.ImageID(image_series_index)}")
    logging.info(f"{images.metadata.ImageName(image_series_index)}")
    images.series = image_series_index
    images.bundle_axes = 'zyxc'
    glomeruli_view = images[0][..., channel_glomeruli]
    podocytes_view = images[0][..., channel_podocytes]

    ground_truth_df = get_marker_coords(xml_tree, 2)
    spatial_columns = ['MarkerZ', 'MarkerY', 'MarkerX']
    gt_image = ground_truth_image(ground_truth_df[spatial_columns].values,
                                  glomeruli_view.shape)

    glomeruli_labels = preprocess_glomeruli(glomeruli_view,
                                            min_glom_diameter,
                                            max_glom_diameter)
    glom_regions = filter_by_size(glomeruli_labels,
                                  min_glom_diameter,
                                  max_glom_diameter)
    logging.info(f"{len(glom_regions)} glomeruli identified.")
    for glom in glom_regions:
        bbox = glom.bbox
        cropping_margin = 10  # pixels
        centroid_offset = tuple(bbox[dim] - cropping_margin
                                for dim in range(podocytes_view.ndim))
        glom_subvolume = crop_region_of_interest(glomeruli_labels,
                                                 bbox,
                                                 margin=cropping_margin,
                                                 pad_mode='zeros')
        gt_subvolume = crop_region_of_interest(gt_image,
                                               bbox,
                                               margin=cropping_margin,
                                               pad_mode='zeros')
        if np.sum(gt_subvolume) == 0:
            # the counts were not from this glomerulus
            continue
        logging.info(f"Glom label: {glom.label}")
        logging.info(f"Glom centroid: {glom.centroid}")
        logging.info(f"Glom bbox: {glom.bbox}")
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
        seeds = blob_dog_image(blobs, podo_subvolume.shape)
        seeds[0] = np.max(seeds) + 1
        pdocyte_gradient = gradient_of_image(podo_subvolume)
        wshed = watershed(pdocyte_gradient, label(seeds))
        wshed[wshed == np.max(seeds)] = 0
        ground_truth_bool = gt_subvolume.astype(np.bool)

        ground_truth_podocyte_number = len(ground_truth_df)
        podocyte_number_found = len(set(wshed.ravel())) - 1
        logging.info(f"ground_truth_podocyte_number: {ground_truth_podocyte_number}")
        logging.info(f"podocyte_number_found: {podocyte_number_found}")

        found_label_set = set(wshed.ravel())
        found_label_set.remove(0)  # excludes zero label
        ground_truth_label_set = set(gt_image.ravel())
        ground_truth_label_set.remove(0)  # excludes zero label

        result = wshed[ground_truth_bool]

        click_but_no_label = list(set(range(1, ground_truth_podocyte_number + 1)) - set(wshed.ravel()))
        labels_with_zero_clicks = set(wshed.ravel()) - set(result)
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
        io.imsave(os.path.join(output_dir, "podo_subvolume.tif"), podo_subvolume.astype(np.float32))
        io.imsave(os.path.join(output_dir, "wshed.tif"), wshed.astype(np.int32))
        io.imsave(os.path.join(output_dir, "glom_subvolume.tif"), glom_subvolume.astype(np.int32))
        io.imsave(os.path.join(output_dir, "gt_subvolume.tif"), gt_subvolume.astype(np.int32))
        io.imsave(os.path.join(output_dir, "mask.tif"), mask.astype(np.int8))
        logging.info("Program finished.")


def match_names(images, xml_image_name, basename):
    n_image_series = images.metadata.ImageCount()
    for i in range(n_image_series):
        images.series = i
        name = images.metadata.ImageName(i)
        full_name = basename + " - " + name
        if xml_image_name == full_name:
            return i
    # if no match found
    return None


def get_marker_coords(tree, n_channels):
    """Parse CellCounter xml"""
    df = pd.DataFrame()
    image_name = tree.find('.//Image_Filename').text
    for marker in tree.findall('.//Marker'):
        x_coord = int(marker.find('MarkerX').text)
        y_coord = int(marker.find('MarkerY').text)
        z_coord = np.floor(int(marker.find('MarkerZ').text) / n_channels)
        contents = {'Image_Filename': image_name,
                    'MarkerX': x_coord,
                    'MarkerY': y_coord,
                    'MarkerZ': z_coord}
        df = df.append(contents, ignore_index=True)
    return df


def ground_truth_image(ground_truth_coords, image_shape):
    """Create label image where pixels labelled with int > 0 match
       coordinates from skimage blob_dog/blob_log/... function."""
    image = np.zeros(image_shape).astype(np.int32)
    for i, gt_coord in enumerate(ground_truth_coords):
        coord = [slice(int(gt_coord[dim]), int(gt_coord[dim]) + 1, 1)
                 for dim in range(image.ndim)]
        image[coord] = i + 1  # only background pixels labelled zero.
    return image


def fix_whitespace(xml_string):
    """Format according to the weird whitespace
       conventions that CellCounter uses."""
    # add the CellCounter header line
    formatted_xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_string
    # change whitespace
    # children
    formatted_xml = formatted_xml.replace("  <Image_Properties>",
                                          " <Image_Properties>"
                                          )
    formatted_xml = formatted_xml.replace("  </Image_Properties>",
                                          " </Image_Properties>"
                                          )
    formatted_xml = formatted_xml.replace("  <Marker_Data>",
                                          " <Marker_Data>"
                                          )
    formatted_xml = formatted_xml.replace("  </Marker_Data>",
                                          " </Marker_Data>"
                                          )
    # grandchildren
    formatted_xml = formatted_xml.replace("    <Image_Filename>",
                                          "     <Image_Filename>"
                                          )
    formatted_xml = formatted_xml.replace("    <Current_Type>",
                                          "     <Current_Type>"
                                          )
    formatted_xml = formatted_xml.replace("    <Marker_Type>",
                                          "     <Marker_Type>"
                                          )
    formatted_xml = formatted_xml.replace("    </Marker_Type>",
                                          "     </Marker_Type>"
                                          )
    # great grandchildren
    formatted_xml = formatted_xml.replace("      <Marker>",
                                          "         <Marker>"
                                          )
    formatted_xml = formatted_xml.replace("      </Marker>",
                                          "         </Marker>"
                                          )
    # great great grandchildren
    formatted_xml = formatted_xml.replace("        <MarkerX>",
                                          "             <MarkerX>"
                                          )
    formatted_xml = formatted_xml.replace("        </MarkerX>",
                                          "             </MarkerX>"
                                          )
    formatted_xml = formatted_xml.replace("        <MarkerY>",
                                          "             <MarkerY>"
                                          )
    formatted_xml = formatted_xml.replace("        </MarkerY>",
                                          "             </MarkerY>"
                                          )
    formatted_xml = formatted_xml.replace("        <MarkerZ>",
                                          "             <MarkerZ>"
                                          )
    formatted_xml = formatted_xml.replace("        </MarkerZ>",
                                          "             </MarkerZ>"
                                          )
    return formatted_xml


def generate_xml(coord_df):
    # root
    new_xml = etree.Element('CellCounter_Marker_File')
    # Image Properties
    Image_Properties = etree.Element('Image_Properties')
    new_xml.append(Image_Properties)
    Image_Filename = etree.Element('Image_Filename')
    Image_Filename.text = str(coord_df['Image_Filename'][0])
    Image_Properties.append(Image_Filename)
    # Marker Data
    Marker_Data = etree.Element('Marker_Data')
    new_xml.append(Marker_Data)
    Current_Type = etree.Element('Current_Type')
    Current_Type.text = '0'
    Marker_Data.append(Current_Type)
    # marker types
    Marker_Type = etree.Element('Marker_Type')
    Marker_Data.append(Marker_Type)
    n_marker_types = 9
    for i in range(1, n_marker_types+1):
        Type = etree.Element('Type')
        Type.text = str(i)
        Marker_Type.append(Type)
    for index, row in coord_df.iterrows():
        Marker = etree.Element('Marker')
        Marker_Type.append(Marker)
        MarkerX = etree.Element('MarkerX')
        MarkerY = etree.Element('MarkerY')
        MarkerZ = etree.Element('MarkerZ')
        MarkerX.text = str(int(row['MarkerX']))
        MarkerY.text = str(int(row['MarkerY']))
        MarkerZ.text = str(int(row['MarkerZ']))
        Marker.append(MarkerX)
        Marker.append(MarkerY)
        Marker.append(MarkerZ)
    # pretty string
    xml_string = etree.tostring(new_xml, pretty_print=True)
    formatted_xml_string = fix_whitespace(xml_string)
    return formatted_xml_string


if __name__=='__main__':
    main()
