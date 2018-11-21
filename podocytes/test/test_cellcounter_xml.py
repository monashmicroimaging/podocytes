import os
import argparse
import xml.etree.ElementTree as ET

from podocytes.cellcounter_xml import main, marker_coords


class TestMarkerXML(object):
    def test_main(self):
        input_directory = os.path.join(os.path.dirname(__file__),
                                       'testdata')
        output_directory = os.path.join(os.path.dirname(__file__),
                                        'output/dir')
        args = argparse.Namespace(input_directory=input_directory,
                                  output_directory=output_directory,
                                  number_of_image_channels=2)
        counts = main(args)
        output = counts['n_podocytes'].iloc[0]
        expected = 48
        os.remove(os.path.join(output_directory,
                               'number_of_podocytes_from_markers.csv'))
        assert output == expected

    def test_marker_coords(self):
        fname = 'testdata/CellCounter_51715_glom6.xml'
        xml_filename = os.path.join(os.path.dirname(__file__), fname)
        xml_tree = ET.parse(xml_filename)
        markers = marker_coords(xml_tree, 2)
        expected = 48
        assert len(markers) == expected
