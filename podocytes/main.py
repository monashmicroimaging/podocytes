import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('wxagg')
from skimage import io
from . import lifio
import matplotlib.pyplot as plt

from gooey.python_bindings.gooey_decorator import Gooey as gooey
from gooey.python_bindings.gooey_parser import GooeyParser

from skimage.util import invert
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import ball, watershed, binary_closing, binary_dilation
from skimage.measure import label, regionprops
from skimage.feature import blob_dog

# Fix from http://chriskiehl.com/article/packaging-gooey-with-pyinstaller/
# Commented out because on py3.6 I get the error "can't have unbuffered TextIO"
# nonbuffered_stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
# sys.stdout = nonbuffered_stdout


__DESCR__ = ('Load, segment, count, and measure glomeruli and podocytes in '
             'fluorescence images.')


@gooey(default_size=(640, 480),
       image_dir=os.path.join(os.path.dirname(__file__), 'app-images'),
       navigation='TABBED')
def main():
    parser = GooeyParser(prog='Podocyte Profiler', description=__DESCR__)
    parser.add_argument('input_files', help='Files to process',
                        widget='MultiFileChooser', nargs='+')
    parser.add_argument('output_folder', help='Output location',
                        widget='DirChooser')
    parser.add_argument('Glomeruli channel number',
                        help="Glomeruli fluorescence image channel")
    parser.add_argument('Podocyte channel number',
                        help="Podocyte fluorescence image channel")
    parser.add_argument('Minimum glomerular diameter (microns)',
                        help="Minimum glomerular diameter (microns)")
    parser.add_argument('Maximum glomerular diameter (microns)',
                        help="Maximum glomerular diameter (microns)")
    args = parser.parse_args()
    import pdb; pdb.set_trace()
    lifio.start()
    print('hooray!')
    summary_statistics = pd.DataFrame()
    detailed_statistics = pd.DataFrame()
    for filename in input_files:
        image_series = lifio.series_iterator(filename)
        names, _, resolutions = lifio.metadata(filename)
        for image_series_number, image in enumerate(image_series):
            voxel_dimensions = resolutions[image_series_number]
            results = process_image(image,
                                    voxel_dimensions,
                                    CHANNEL_GLOMERULI,
                                    CHANNEL_PODOCYTES,
                                    GLOMERULUS_DIAMETER_MINIMUM,
                                    GLOMERULUS_DIAMETER_MAXIMUM)
            results["image_series_number"] = image_series_number
            results["image_series_name"] = names[image_series_number]
            results["image_filename"] = filename
            detailed_statistics = detailed_statistics.append(df_results)
            # Write results to file as you go
            detailed_statistics.to_csv(os.path.join(output_folder, "Podocyte_detailed_statistics.csv"))
    # Optional creation of summary statistics
    summary_columns = ["image_filename",
                       "image_series_name",
                       "image_series_number",
                       "glomeruli_label_number",
                       "glomeruli_voxel_number",
                       "glomeruli_volume",
                       "glomeruli_equivalent_diameter_voxels",
                       "glomeruli_centroid",
                       "avg_podocyte_voxel_number",
                       "avg_podocyte_volume",
                       "avg_podocyte_equivalent_diameter_voxels",
                       "avg_podocyte_eccentricity",
                       "avg_podocyte_major_axis_length",
                       "avg_podocyte_minor_axis_length",
                       "avg_podocyte_mean_intensity",
                       "avg_podocyte_max_intensity",
                       "avg_podocyte_min_intensity"]
    summary_statistics = detailed_statistics[summary_columns].drop_duplicates()
    summary_statistics.to_csv(os.path.join(output_folder, "Podocyte_summary_statistics.csv"))
    lifio.done()


def process_image(input_image, voxel_dimensions,
                  CHANNEL_GLOMERULI, CHANNEL_PODOCYTES,
                  GLOMERULUS_DIAMETER_MINIMUM, GLOMERULUS_DIAMETER_MAXIMUM):
    """Process single image volume."""
    glomeruli_view = input_image[:, CHANNEL_GLOMERULI, :, :]
    podocytes_view = input_image[:, CHANNEL_PODOCYTES, :, :]
    # Denoise images with small gaussian blur
    voxel_volume = np.prod(voxel_dimensions)
    sigma = np.divide(voxel_dimensions[-1], voxel_dimensions) # non-isotropic
    glomeruli_view = gaussian(glomeruli_view, sigma=sigma)
    podocytes_view = gaussian(podocytes_view, sigma=sigma)
    # Find the glomeruli
    threshold_glomeruli = threshold_otsu(glomeruli_view)
    glomeruli_regions = find_glomeruli(glomeruli_view,
                                       threshold_glomeruli,
                                       GLOMERULUS_DIAMETER_MINIMUM,
                                       GLOMERULUS_DIAMETER_MAXIMUM)
    for glom in glomeruli_regions:
        df = pd.DataFrame()
        podocyte_regions = find_podocytes(podocytes_view, glom)
        for pod in podocyte_regions:
            # Add interesting statistics to the dataframe
            contents = {"glomeruli_label_number": glom.label,
                        "glomeruli_voxel_number": glom.filled_area,
                        "glomeruli_volume": (glom.filled_area * voxel_volume),
                        "glomeruli_equivalent_diameter_voxels":
                            glom.equivalent_diameter,
                        "glomeruli_centroid": glom.centroid,
                        "podocyte_label_number": pod.label,
                        "podocyte_voxel_number": pod.area,
                        "podocyte_volume": (pod.area * voxel_volume),
                        "podocyte_equivalent_diameter_voxels":
                            pod.equivalent_diameter,
                        "podocyte_centroid": pod.centroid,
                        "podocyte_eccentricity": pod.eccentricity,
                        "podocyte_major_axis_length": pod.major_axis_length,
                        "podocyte_minor_axis_length": pod.minor_axis_length,
                        "podocyte_mean_intensity": pod.mean_intensity,
                        "podocyte_max_intensity": pod.max_intensity,
                        "podocyte_min_intensity": pod.min_intensity}
            # Add individual podocyte statistics to dataframe
            df = df.append(contents, ignore_index=True)
        # Add summary statistics (average of all podocytes in glomerulus)
        df["avg_podocyte_voxel_number"] = np.mean(df["podocyte_voxel_number"])
        df["avg_podocyte_volume"] = np.mean(df["podocyte_volume"])
        df["avg_podocyte_equivalent_diameter_voxels"] = np.mean(df["podocyte_equivalent_diameter_voxels"])
        df["avg_podocyte_eccentricity"] = np.mean(df["podocyte_eccentricity"])
        df["avg_podocyte_major_axis_length"] = np.mean(df["podocyte_major_axis_length"])
        df["avg_podocyte_minor_axis_length"] = np.mean(df["podocyte_minor_axis_length"])
        df["avg_podocyte_mean_intensity"] = np.mean(df["podocyte_mean_intensity"])
        df["avg_podocyte_max_intensity"] = np.mean(df["podocyte_max_intensity"])
        df["avg_podocyte_min_intensity"] = np.mean(df["podocyte_min_intensity"])
        return df


def find_glomeruli(glomeruli_image, threshold, min_diameter, max_diameter):
    """Identify glomeruli in the image volume and yield those regions."""
    label_image = label(glomeruli_image > threshold)
    for region in regionprops(label_image):
        if ((region.equivalent_diameter >= min_diameter) and
                (region.equivalent_diameter <= max_diameter)):
            yield region


def find_podocytes(podocyte_image, glomeruli_region,
                   min_sigma=1, max_sigma=4, dog_threshold=0.17):
    """Identify podocytes in the image volume."""
    bbox = glomeruli_region.bbox  # bounding box coordinates
    image_roi = crop_region_of_interest(podocyte_image, bbox, margin=10)
    threshold = threshold_otsu(image_roi)
    mask = image_roi > threshold
    blobs = blob_dog(image_roi,
                     min_sigma=min_sigma,
                     max_sigma=max_sigma,
                     threshold=dog_threshold)
    seeds = blob_dog_image(blobs, image_roi.shape)
    wshed = watershed(invert(image_roi), label(seeds), mask=mask)
    regions = regionprops(wshed, intensity_image=image_roi)
    return regions


def crop_region_of_interest(image, bbox, margin=0, pad_mode='mean'):
    """Return cropped region of interest, with border padding.

    Parameters
    ----------
    image : (N, M) ndarray
        The input image.
    bbox : tuple
        Bounding box coordinates as tuple.
        Returned from scikit-image regionprops bbox attribute. Format is:
        3D example (min_pln, min_row, min_col, max_pln, max_row, max_col)
        2D example (min_row, min_col, max_row, max_col)
        Pixels belonging to the bounding box are in the half-open interval.
    margin : int, optional
        How many pixels to increase the size of the bounding box by.
        If this margin exceeds the input image array bounds,
        then the output image is padded.
    pad_mode : string, optional
        Type of border padding to use. Is either 'mean' (default) or 'zeros'.
    Returns
    -------
    roi_image : (N, M) ndarray
        The cropped output array.
    """
    ndims = image.ndim
    max_image_size = np.array([np.size(image, dim) for dim in range(ndims)])
    bbox_min_plus_margin = np.array([coord - margin for coord in bbox[:ndims]])
    bbox_max_plus_margin = np.array([coord + margin for coord in bbox[ndims:]])
    image_min_coords = bbox_min_plus_margin.clip(min=0)
    image_max_coords = bbox_max_plus_margin.clip(max=max_image_size)
    max_roi_size = np.array([abs(bbox_max_plus_margin[dim] -
                                 bbox_min_plus_margin[dim])
                             for dim in range(ndims)])
    roi_min_coord = abs(image_min_coords - bbox_min_plus_margin)
    roi_max_coord = max_roi_size - abs(bbox_max_plus_margin - image_max_coords)
    image_slicer = tuple(slice(image_min_coords[dim], image_max_coords[dim], 1)
                         for dim in range(ndims))
    roi_slicer = tuple(slice(roi_min_coord[dim], roi_max_coord[dim], 1)
                       for dim in range(ndims))
    if pad_mode == 'zeros':
        roi_image = np.zeros(max_roi_size)
    elif pad_mode == 'mean':
        roi_image = np.ones(max_roi_size) * np.mean(image[image_slicer])
    else:
        raise ValueError("'pad_mode' keyword argument unrecognized.")
    roi_image[roi_slicer] = image[image_slicer]
    return roi_image


def blob_dog_image(blobs, image_shape):
    """Create boolean image where pixels labelled True
      match coordinates returned from skimage blob_dog() function."""
    blob_map = np.zeros(image_shape).astype(np.bool)
    for blob in blobs:
        blob_map[int(blob[0]), int(blob[1]), int(blob[2])] = True
    return blob_map


if __name__ == '__main__':
    main()
