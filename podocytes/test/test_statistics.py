import os
import pandas as pd

from podocytes.statistics import (glom_statistics,
                                  podocyte_statistics,
                                  podocyte_avg_statistics,
                                  summarize_statistics)


def test_podocyte_avg_statistics():
    df = pd.DataFrame({'podocyte_voxel_number': [100, 200, 300],
                       'podocyte_volume': [30, 40, 50],
                       'podocyte_equiv_diam_pixels': [7, 8, 9]})
    output = podocyte_avg_statistics(df)
    expected = pd.DataFrame({'podocyte_voxel_number': [100, 200, 300],
                             'podocyte_volume': [30, 40, 50],
                             'podocyte_equiv_diam_pixels': [7, 8, 9],
                             'avg_podocyte_voxel_number': [200, 200, 200],
                             'avg_podocyte_volume': [40, 40, 40],
                             'avg_podocyte_equiv_diam_pixels': [8, 8, 8]})
    assert output.all().all() == expected.all().all()


def test_summarize_statistics():
    input_filename = os.path.join(os.path.dirname(__file__),
        'testdata/csv/Podocyte_detailed_stats_12-Oct-2018_12-02PM.csv')
    output_filename = os.path.join(os.path.dirname(__file__),
        'output/dir/test_summary_stats.csv')
    expected_filename = os.path.join(os.path.dirname(__file__),
        'testdata/csv/Podocyte_summary_stats_12-Oct-2018_12-02PM.csv')
    detailed_stats = pd.read_csv(input_filename)
    output = summarize_statistics(detailed_stats, output_filename)
    expected = pd.read_csv(expected_filename)
    os.remove(output_filename)
    assert output.all().all() == expected.all().all()
