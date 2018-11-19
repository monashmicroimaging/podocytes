import os
import logging

import numpy as np
import pandas as pd
from gooey.python_bindings.gooey_decorator import Gooey as gooey
from gooey.python_bindings.gooey_parser import GooeyParser
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from podocytes.util import log_file_begins, find_files, marker_coords


def main(args):
    counts = process_folder(args)
    counts.to_csv(os.path.join(args.output_directory, "number_of_podocytes_from_markers.csv"))
    return counts


__DESCR__ = ('Count the number of markers in CellCounter xml files.')
@gooey(default_size=(800, 700),
       image_dir=os.path.join(os.path.dirname(__file__), '../app-images'),
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
    parser.add_argument('number_of_image_channels',
                        help='Total number of color channels in image.',
                        type=int, default=2)
    return parser


def process_folder(args):
    """
    Parameters
    ----------
    args : user input arguments

    Returns
    -------
    counts :
    """
    marker_files = find_files(args.input_directory, '.xml')
    contents = []
    for xml_filename in marker_files:
        mouse = os.path.basename(os.path.dirname(xml_filename))
        glom_id = os.path.splitext(os.path.basename(xml_filename))[0][-2:]
        xml_tree = ET.parse(xml_filename)
        xml_image_name = xml_tree.find('.//Image_Filename').text
        markers = marker_coords(xml_tree, args.number_of_image_channels)
        n_podocytes = len(markers)
        logging.info(f"{n_podocytes} markers counted from file: {xml_filename}")
        contents.append([xml_filename, xml_image_name, mouse, glom_id, n_podocytes])
    counts = pd.DataFrame(contents, columns=['filename', 'xml_image_name', 'mouse', 'glom_id', 'n_podocytes'])
    return counts


def marker_coords(tree, n_channels):
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


if __name__ == "__main__":
    parser = configure_parser()
    args = parser.parse_args()
    log_file_begins(args)
    main(args)
