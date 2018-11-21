import os
import argparse
import xml.etree.ElementTree as ET

from podocytes.cellcounter_xml import main, marker_coords


def test_main(tmpdir):
    input_directory = os.path.join(os.path.dirname(__file__),
                                   'testdata')
    args = argparse.Namespace(input_directory=input_directory,
                              output_directory=tmpdir,
                              number_of_image_channels=2)
    counts = main(args)
    output = counts['n_podocytes'].iloc[0]
    expected = 48
    assert output == expected


def test_marker_coords():
    fname = 'testdata/CellCounter_51715_glom6.xml'
    xml_filename = os.path.join(os.path.dirname(__file__), fname)
    xml_tree = ET.parse(xml_filename)
    markers = marker_coords(xml_tree, 2)
    expected = 48
    assert len(markers) == expected
