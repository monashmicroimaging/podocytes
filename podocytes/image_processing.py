import numpy as np

from skimage.util import invert
from skimage.filters import threshold_yen, gaussian
from skimage.morphology import ball, watershed, binary_closing, binary_dilation
from skimage.measure import label, regionprops
from skimage.feature import blob_dog


__all__ = ['crop_region_of_interest',
           'denoise_image',
           'filter_by_size',
           'find_glomeruli',
           'find_podocytes',
           'gradient_of_image',
           'marker_controlled_watershed',
           'markers_from_blob_coords',
           'ground_truth_image']


def crop_region_of_interest(image, bbox, margin=0, pad_mode='mean'):
    """Return cropped region of interest, with border padding.

    Parameters
    ----------
    image : 3D ndarray
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
    roi_image : 3D ndarray
        The cropped output array.
    """
    ndims = image.ndim
    max_image_size = np.array([np.size(image, dim) for dim in range(ndims)])
    bbox_min_plus_margin = np.array([coord - margin for coord in bbox[:ndims]])
    bbox_max_plus_margin = np.array([coord + margin for coord in bbox[ndims:]])
    image_min_coords = bbox_min_plus_margin.clip(min=0)
    image_max_coords = bbox_max_plus_margin.clip(max=max_image_size)
    max_roi_size = np.array([
        abs(bbox_max_plus_margin[dim] - bbox_min_plus_margin[dim])
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


def denoise_image(image):
    """Denoise images with a slight gaussian blur.

    Parameters
    ----------
    image : 3D ndarray
        Original image data from a single fluorescence channel.

    Returns
    -------
    denoised : 3D ndarray
        Image denoised by slight gaussian blur.
    """
    xy_pixel_size = image.metadata['mpp']
    z_pixel_size = image.metadata['mppZ']
    voxel_dimensions = []
    for i in image.metadata['axes']:
        if i == 'x' or i == 'y':
            voxel_dimensions.append(xy_pixel_size)
        elif i == 'z':
            voxel_dimensions.append(z_pixel_size)
    sigma = np.divide(xy_pixel_size, voxel_dimensions)  # non-isotropic
    denoised = gaussian(image, sigma=sigma)
    return denoised


def filter_by_size(label_image, min_diameter, max_diameter):
    """Identify objects within a certain size range & return those regions.

    Uses the equivalent_diameter attribute of regionprops to check size.

    Parameters
    ----------
    label_image : 3D ndarray
        Label image
    min_diameter : float
        Minimum expected size (equivalent diameter of labelled voxels)
        Uses the equivalent_diameter attribute of scikit-image regionprops.
    max_diameter : float
        Maximum expected size (equivalent diameter of labelled voxels)
        Uses the equivalent_diameter attribute of scikit-image regionprops.

    Returns
    -------
    regions : list of RegionProperties
        RegionProperties from scikit-image.
    """
    regions = []
    for region in regionprops(label_image):
        if region.equivalent_diameter >= min_diameter:
            if region.equivalent_diameter <= max_diameter:
                regions.append(region)
    return regions


def find_glomeruli(glomeruli_view):
    """Preprocess glomeruli channel image, return labelled glomeruli image.

    Parameters
    ----------
    glomeruli_view : 3D ndarray
        Image array of glomeruli fluorescence channel.

    Returns
    -------
    label_image : 3D ndarray
        Label image identifying fluorescence regions in glomeruli channel.
    """
    glomeruli_view = denoise_image(glomeruli_view)
    threshold = threshold_yen(glomeruli_view)
    label_image = label(glomeruli_view > threshold)
    return label_image


def find_podocytes(podocyte_image, glomeruli_region,
                   min_sigma=1, max_sigma=4, dog_threshold=0.17,
                   cropping_margin=10):
    """Identify podocytes in the image volume.

    Parameters
    ----------
    podocyte_image : 3D ndarray
        Image of podocyte fluorescence.
    glomeruli_region : RegionProperties
        Single glomeruli region, found with scikit-image regionprops.
    min_sigma : float, optional
        Minimum sigma to find podocyte blobs using difference of gaussians.
    max_sigma : float, optional
        Maximum sigma to find podocyte blobs using difference of gaussians.
    dog_threshold : float, optional
        Threshold value for difference of gaussian blob finding.
    cropping_margin : int, optional
        How many pixels for the margin around each glomerulus when cropping.

    Returns
    -------
    regions : List of RegionProperties
        Region properties for podocytes identified.
    centroid_offset : tuple of int
        Coordinate offset of glomeruli subvolume in image.
    wshed : 3D ndarray
        Watershed image showing podoyctes.
    """
    bbox = glomeruli_region.bbox  # bounding box coordinates
    centroid_offset = tuple(bbox[dim] - cropping_margin
                            for dim in range(podocyte_image.ndim))
    image_roi = crop_region_of_interest(podocyte_image, bbox,
                                        margin=cropping_margin)
    blobs = blob_dog(image_roi,
                     min_sigma=min_sigma,
                     max_sigma=max_sigma,
                     threshold=dog_threshold)
    wshed = marker_controlled_watershed(image_roi, blobs)
    regions = regionprops(wshed, intensity_image=image_roi)
    return (regions, centroid_offset, wshed)


def gradient_of_image(image):
    """Take the maximum absolute gradient of the image in all directions."""
    grad = np.gradient(image)  # gradients for individual directions
    grad = np.stack(grad, axis=-1)  # from list of arrays to single numpy array
    gradient_image = np.sum(abs(grad), axis=-1)
    return gradient_image


def marker_controlled_watershed(grayscale_image, marker_coords):
    """Returns the watershed result given a grayscale image and marker seeds.

    Parameters
    ----------
    grayscale_image : 3D ndarray
        Input image to apply watershed on.
    marker_coords : 3D ndarray
        Array where the first consecutive elements in each row
        are the spatial coordinates of the markers.

    Returns
    -------
    wshed : 3D ndarray
        Label image of watershed results.
    """
    gradient_image = gradient_of_image(grayscale_image)
    seeds = markers_from_blob_coords(marker_coords, grayscale_image.shape)
    wshed = watershed(gradient_image, seeds)
    wshed[wshed == np.max(seeds)] = 0  # set background area to zero
    return wshed


def markers_from_blob_coords(blobs, image_shape):
    """Make watershed markers from scikit-image blob_dog coordinates.

    Parameters
    ----------
    blobs : (n, image.ndim + 1) ndarray
        Input is the output of skimage.feature.blog_dog()
    image_shape : tuple of int
        Shape of image array used to generate input parameter blobs.

    Returns
    -------
    markers : 2D or 3D ndarray
        Boolean array shaped like image_shape,
        with blob coordinate locations represented by True values.
    """
    markers = np.zeros(image_shape, dtype=bool)
    markers[tuple(blobs[:, :-1].T.astype(int))] = True
    # This assumes the first voxel is part of the background
    markers.ravel()[0] = True  # must have seed for background area
    markers = label(markers)
    return markers


def ground_truth_image(ground_truth_coords, image_shape):
    """Label image from coordinates in CellCounter xml marker file.

    Creates a label image where pixels labelled with int > 0 match
    coordinates from xml cellcounter marker file.
    Can also be used with coords from skimage blob_dog/blob_log/blob_doh.

    Parameters
    ----------
    ground_truth_coords :  array where first columns are x, y, and z coords.
    image_shape : shape of image, eg: img_array.shape

    Returns
    -------
    image : label image
    """
    image = np.zeros(image_shape).astype(np.int32)
    for i, gt_coord in enumerate(ground_truth_coords):
        coord = [slice(int(gt_coord[dim]), int(gt_coord[dim]) + 1, 1)
                 for dim in range(image.ndim)]
        image[coord] = i + 1  # only background pixels labelled zero.
    return image
