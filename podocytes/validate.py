import os
import sys
import time
import logging
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


def main():
    pass


def match_names(xml_image_name, basename, images):
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


def generate_cellcounter_xml(df, ):
    """Create xml file able to be read by CellCounter."""
    pass


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
    formatted_xml = formatted_xml.replace("  <Image_Properties>",  " <Image_Properties>")
    formatted_xml = formatted_xml.replace("  </Image_Properties>", " </Image_Properties>")
    formatted_xml = formatted_xml.replace("  <Marker_Data>",  " <Marker_Data>")
    formatted_xml = formatted_xml.replace("  </Marker_Data>", " </Marker_Data>")
    # grandchildren
    formatted_xml = formatted_xml.replace("    <Image_Filename>", "     <Image_Filename>")
    formatted_xml = formatted_xml.replace("    <Current_Type>", "     <Current_Type>")
    formatted_xml = formatted_xml.replace("    <Marker_Type>",  "     <Marker_Type>")
    formatted_xml = formatted_xml.replace("    </Marker_Type>", "     </Marker_Type>")
    # great grandchildren
    formatted_xml = formatted_xml.replace("      <Marker>",  "         <Marker>")
    formatted_xml = formatted_xml.replace("      </Marker>", "         </Marker>")
    # great great grandchildren
    formatted_xml = formatted_xml.replace("        <MarkerX>",  "             <MarkerX>")
    formatted_xml = formatted_xml.replace("        </MarkerX>", "             </MarkerX>")
    formatted_xml = formatted_xml.replace("        <MarkerY>",  "             <MarkerY>")
    formatted_xml = formatted_xml.replace("        </MarkerY>", "             </MarkerY>")
    formatted_xml = formatted_xml.replace("        <MarkerZ>",  "             <MarkerZ>")
    formatted_xml = formatted_xml.replace("        </MarkerZ>", "             </MarkerZ>")
    return formatted_xml


# generate xml
# you need to add this line somehow to the top
# <?xml version="1.0" encoding="UTF-8"?>
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
    import pdb; pdb.set_trace()
    main()
