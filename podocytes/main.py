import os
import sys

import numpy as np
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
nonbuffered_stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stdout = nonbuffered_stdout


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
    args = parser.parse_args()
    lifio.start()
    print('hooray!')
    lifio.done()


def process_image(input_image, CHANNEL_GLOMERULI, CHANNEL_PODOCYTES,
                  GLOMERULUS_DIAMETER_MINIMUM, GLOMERULUS_DIAMETER_MAXIMUM,
                  PIXEL_SIZE_X, PIXEL_SIZE_Y, PIXEL_SIZE_Z):
    """Process single image volume."""
    glomeruli_view = input_image[:, CHANNEL_GLOMERULI, :, :]
    podocytes_view = input_image[:, CHANNEL_PODOCYTES, :, :]
    ndims = glomeruli_view.ndim  # number of spatial dimensions
    # Denoise images with small gaussian blur
    voxel_volume = PIXEL_SIZE_X * PIXEL_SIZE_Y * PIXEL_SIZE_Z
    ratio_x = 1.  # ratio of voxel dimensions, with respect to x dimension.
    ratio_y = PIXEL_SIZE_X/PIXEL_SIZE_Y
    ratio_z = PIXEL_SIZE_X/PIXEL_SIZE_Z
    sigma = [ratio_x, ratio_y, ratio_z]  # usually sigma = [1., 1., 0.5]
    glomeruli_view = gaussian(glomeruli_view, sigma=sigma)
    podocytes_view = gaussian(podocytes_view, sigma=sigma)
    # Find the glomeruli
    threshold_glomeruli = threshold_otsu(glomeruli_view)
    glomeruli_regions = find_glomeruli(glomeruli_view,
                                       threshold_glomeruli,
                                       GLOMERULUS_DIAMETER_MINIMUM,
                                       GLOMERULUS_DIAMETER_MAXIMUM)
    for glom in glomeruli_regions:
        glom_volume_in_voxels = glom.filled_area
        glom_volume_realspace = voxel_volume * glom_volume_in_voxels
        podocyte_regions = find_podocytes(podocytes_view, glom, )
        for pod in podocyte_regions:
            # Get interesting statistics about the data & output as csv
            podocyte_volume_in_voxels = pod.area
            podocyte_volume_realspace = voxel_volume * podocyte_volume_in_voxels
            #pod.centroid
            #pod.weighted_centroid  # weighted_centroid : array, Centroid coordinate tuple (row, col) weighted with intensity image.
            #pod.eccentricity
            #pod.equivalent_diameter  # ?
            #pod.max_intensity
            #pod.mean_intensity
            #pod.min_intensity
            #pod.major_axis_length
            #pod.minor_axis_length


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


def crop_region_of_interest(image, bbox, margin=0):
    """Return cropped region of interest, with optional mean border padding."""
    ndims = image.ndim
    image_max_size = [np.size(image, dim) for dim in range(ndims)]
    min_coords = [bbox[dim] - margin for dim in range(ndims)]
    max_coords = [bbox[ndims + dim] + margin for dim in range(ndims)]
    # use mean intensity padding at borders if array out of bounds
    # changing the approach here
    #return image_roi


def blob_dog_image(blobs, image_shape):
    """Create boolean image where pixels labelled True
      match coordinates returned from skimage blob_dog() function."""
    blob_map = np.zeros(image_shape).astype(np.bool)
    for blob in blobs:
        blob_map[int(blob[0]), int(blob[1]), int(blob[2])] = True
    return blob_map


if __name__ == '__main__':
    main()
