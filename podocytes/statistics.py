import logging

import numpy as np
import pandas as pd


def glom_statistics(df, glom, glom_index, voxel_volume):
    """Add glomerulus information to podocyte statistics for a single glom.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing values for podocytes.
    glom : RegionProperties
         Glomerulus region properties (from scikit-image regionprops).
    glom_index : int
        Integer label for glomerulus.
    voxel_volume : float
        Real space volume of a single image voxel.

    Returns
    -------
    df : DataFrame
        Input pandas dataframe now containing additional columns
        containing data about the outer glomerulus.
    """
    df['number_of_podocytes'] = len(df)
    df['podocyte_density'] = len(df) / (glom.filled_area * voxel_volume)
    df['glomeruli_index'] = glom_index
    df['glomeruli_label_number'] = glom.label
    df['glomeruli_voxel_number'] = glom.filled_area
    df['glomeruli_volume'] = (glom.filled_area * voxel_volume)
    df['glomeruli_equiv_diam_pixels'] = glom.equivalent_diameter
    df['glomeruli_centroid_x'] = glom.centroid[2]
    df['glomeruli_centroid_y'] = glom.centroid[1]
    df['glomeruli_centroid_z'] = glom.centroid[0]
    return df


def podocyte_statistics(podocyte_regions, centroid_offset, voxel_volume):
    """Calculate statistics for podocytes.

    Parameters
    ----------
    podocyte_regions : List of RegionProperties
        Region properties for podocytes identified.
    centroid_offset : tuple of int
        Coordinate offset of glomeruli subvolume in image.
    voxel_volume : float
        Real space volume of a single image voxel.

    Returns
    -------
    df : DataFrame or None
        Pandas dataframe containing podocytes statistics,
        or None if no regions matching podocyte criteria were identified.
    """
    column_names = ['podocyte_label_number',
                    'podocyte_voxel_number',
                    'podocyte_volume',
                    'podocyte_equiv_diam_pixels',
                    'podocyte_centroid_x',
                    'podocyte_centroid_y',
                    'podocyte_centroid_z']
    contents_list = []
    for pod in podocyte_regions:
        real_podocyte_centroid = tuple(pod.centroid[dim] +
                                       centroid_offset[dim]
                                       for dim in range(len(pod.centroid)))
        # Add interesting statistics to the dataframe
        # Centroid coords are (x, y, z) and NOT (plane, row, column)
        contents = [pod.label,
                    pod.area,
                    pod.area * voxel_volume,
                    pod.equivalent_diameter,
                    real_podocyte_centroid[2],
                    real_podocyte_centroid[1],
                    real_podocyte_centroid[0]]
        # Add individual podocyte statistics to dataframe
        contents_list.append(contents)
    df = pd.DataFrame(contents_list, columns=column_names)
    return df


def podocyte_avg_statistics(df):
    """Average podocyte statistics per glomerulus.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing podocyte statistics for a single glomerulus

    Returns
    -------
    df : DataFrame
        Pandas dataframe containing additional series with aggregate statistics
        for podocytes per glomerulus, including
        * the average podocyte voxel number
        * the average volume in real space of the podocytes
        * the average equivalent diameter of podocytes (in pixels)
    """
    df['avg_podocyte_voxel_number'] = np.mean(df['podocyte_voxel_number'])
    df['avg_podocyte_volume'] = np.mean(df['podocyte_volume'])
    df['avg_podocyte_equiv_diam_pixels'] = np.mean(df['podocyte_equiv_diam_pixels'])
    return df


def summarize_statistics(detailed_stats, output_filename):
    """Return dataframe with average podocyte statistics per glomerulus.

    Parameters
    ----------
    detailed_stats : DataFrame
        Pandas dataframe containing detailed podocyte statistics.
    output_filename : str
        Filepath to save summarized statistics to.

    Returns
    -------
    summary_stats : DataFrame
        Pandas dataframe containing average podocyte statistics per glomerulus.
    """
    if len(detailed_stats) > 0:
        summary_columns = ['image_filename',
                           'image_series_name',
                           'image_series_num',
                           'glomeruli_index',
                           'glomeruli_label_number',
                           'glomeruli_voxel_number',
                           'glomeruli_volume',
                           'glomeruli_equiv_diam_pixels',
                           'glomeruli_centroid_x',
                           'glomeruli_centroid_y',
                           'glomeruli_centroid_z',
                           'number_of_podocytes',
                           'avg_podocyte_voxel_number',
                           'avg_podocyte_volume',
                           'podocyte_density']
        summary_stats = detailed_stats[summary_columns].drop_duplicates()
        summary_stats.reset_index(drop=True, inplace=True)
        summary_stats.to_csv(output_filename)
        logging.info(f'Saved summary statistics to file: {output_filename}')
        return summary_stats
