import os
import argparse
import xml.etree.ElementTree as ET

import pims
import numpy as np
import pandas as pd

from podocytes import validate


def test_main():
    input_image_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    input_xml_dir = input_image_dir
    output_dir = os.path.join(os.path.dirname(__file__), 'output/dir')
    args = argparse.Namespace(input_directory=input_image_dir,
                              output_directory=output_dir,
                              glomeruli_channel_number=0,
                              podocyte_channel_number=1,
                              minimum_glomerular_diameter=30.0,
                              maximum_glomerular_diameter=300.0,
                              file_extension='.tif',
                              xml_extension='.xml',
                              counts_directory=input_xml_dir)
    output = validate.main(args)
    column_names = ['n_podocytes_ground_truth',
                    'n_podocytes_found',
                    'difference_in_podocyte_number',
                    'gleom_label',
                    'glom_centroid_x',
                    'glom_centroid_y',
                    'glom_centroid_z',
                    'image_filename',
                    'xml_filename']
    expected_image_filename = os.path.join(input_image_dir,
                                           '51715_glom6.tif')
    expected_xml_filename = os.path.join(input_xml_dir,
                                         'CellCounter_51715_glom6.xml')
    expected_content = [[48, 48, 0, 1, 107.91522420803565,
                         86.65583147641911, 31.7051899795548,
                         expected_image_filename, expected_xml_filename]]
    expected = pd.DataFrame(expected_content, columns=column_names)
    os.remove(os.path.join(os.path.dirname(__file__),
                           'output/dir/51715_glom6_glomlabel1.tif'))
    os.remove(os.path.join(output_dir, 'Podocyte_validation_stats.csv'))
    assert output.all().all() == expected.all().all()


def test_validate_image():
    output_dir = os.path.join(os.path.dirname(__file__), 'output/dir')
    args = argparse.Namespace(input_directory='testdata',
                              output_directory=output_dir,
                              glomeruli_channel_number=0,
                              podocyte_channel_number=1,
                              minimum_glomerular_diameter=30.0,
                              maximum_glomerular_diameter=300.0,
                              file_extension='.tif',
                              xml_extension='.xml',
                              counts_directory='testdata')
    # input xml file
    xml_fname = 'testdata/CellCounter_51715_glom6.xml'
    xml_filename = os.path.join(os.path.dirname(__file__), xml_fname)
    xml_tree = ET.parse(xml_filename)
    # input image
    fname = 'testdata/51715_glom6.tif'
    filename = os.path.join(os.path.dirname(__file__), fname)
    images = pims.Bioformats(filename)
    images.bundle_axes = 'zyxc'
    output = validate.validate_image(args, images[0], xml_tree)
    column_names = ['n_podocytes_ground_truth',
                    'n_podocytes_found',
                    'difference_in_podocyte_number',
                    'gleom_label',
                    'glom_centroid_x',
                    'glom_centroid_y',
                    'glom_centroid_z']
    expected_content = [[48, 48, 0, 1,
                         107.91522420803565,
                         86.65583147641911,
                         31.7051899795548]]
    expected = pd.DataFrame(expected_content, columns=column_names)
    os.remove(os.path.join(os.path.dirname(__file__),
                           'output/dir/51715_glom6_glomlabel1.tif'))
    assert output.all().all() == expected.all().all()


def test_count_podocytes_in_label_image():
    label_image = np.zeros((128, 128, 128))
    label_image[20:25, 20:25, 20:25] = 1
    label_image[50:55, 50:55, 50:55] = 2
    label_image[70:75, 70:75, 70:75] = 3
    output = validate.count_podocytes_in_label_image(label_image)
    expected = 3
    assert output == expected


def test_match_filenames_1():
    xml_image_name = '51715_glom6.tif'
    image_filenames = ['/test/testdata/51715_glom6.tif',
                       '/test/testdata/corrupt_image.tif']
    output = validate.match_filenames(image_filenames, xml_image_name)
    expected = '/test/testdata/51715_glom6.tif'
    assert output == expected


def test_match_filenames_2():
    xml_image_name = '51715_glom6.tif'
    image_filenames = ['/test/testdata/not_a_match.tif',
                       '/test/testdata/corrupt_image.tif']
    output = validate.match_filenames(image_filenames, xml_image_name)
    expected = None
    assert output == expected
