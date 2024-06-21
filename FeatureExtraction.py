#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 22:33:05 2024

@author: ernest
"""

import SimpleITK as sitk
from radiomics import featureextractor
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


# Extracting full set of radiomics
def radiomics_extractor(img, msk, binw=25):
    """
    Extract radiomics features from an image and a mask.

    Parameters:
    img (numpy.ndarray): The input image as a numpy array.
    msk (numpy.ndarray): The corresponding mask as a numpy array.
    binw (float): Bin width for radiomics extraction.

    Returns:
    dict: Dictionary of extracted radiomics features.
    """
    # Convert numpy arrays to SimpleITK images
    img_sitk = sitk.GetImageFromArray(img)
    msk_sitk = sitk.GetImageFromArray(msk)
    
    # Initialize the feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=binw)
    
    # Enable all image types and features
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()
    
    # Extract features
    featureVector = extractor.execute(img_sitk, msk_sitk)
    
    return featureVector


def load_nifti_with_orientation(file_path):
    """
    Load a NIfTI file, reorient it to RAS, and return the numpy array and affine.

    Parameters:
    file_path (str): Path to the NIfTI file.

    Returns:
    tuple: Reoriented numpy array and the corresponding affine matrix.
    """
    nii = nib.load(file_path)
    ras_img = nib.as_closest_canonical(nii)
    ras_data = ras_img.get_fdata()
    affine = ras_img.affine
    return ras_data, affine


def BinarizeMask(mask):
    """
    Binarize a segmentation mask to have values 0 or 1 (whole tumor vs background).

    Parameters:
    mask (numpy.ndarray): The input segmentation mask.

    Returns:
    numpy.ndarray: The binarized mask.
    """
    binarized_mask = (mask > 0).astype(np.uint8)
    return binarized_mask


def find_index_slice(binary_mask, orientation='ax', plot=False):
    """
    Find the index of the slice with the maximum tumor area based on the specified orientation,
    plot the areas if plot=True.

    Parameters:
    binary_mask (numpy.ndarray): The binarized segmentation mask in RAS orientation.
    orientation (str): The orientation to consider ('ax' for axial, 'sag' for sagittal, 'cor' for coronal).
    plot (bool): Whether to plot the tumor area across slices.

    Returns:
    int: The index of the slice with the maximum tumor area in the specified orientation.
    """
    if orientation not in ['ax', 'sag', 'cor']:
        raise ValueError("Invalid orientation. Must be one of 'ax', 'sag', 'cor'.")

    # Calculate the tumor area for each slice based on the specified orientation
    if orientation == 'ax':
        num_slices = binary_mask.shape[2]
        areas = [np.sum(binary_mask[:, :, i]) for i in range(num_slices)]
    elif orientation == 'sag':
        num_slices = binary_mask.shape[0]
        areas = [np.sum(binary_mask[i, :, :]) for i in range(num_slices)]
    elif orientation == 'cor':
        num_slices = binary_mask.shape[1]
        areas = [np.sum(binary_mask[:, i, :]) for i in range(num_slices)]

    # Plot the tumor areas if required
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_slices), areas, label='Tumor Area')
        plt.xlabel('Slice Index')
        plt.ylabel('Tumor Area')
        plt.title(f'Tumor Area across {orientation.upper()} Slices')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Find the slice with the maximum tumor area
    max_area_index = np.argmax(areas)
    print(f"Maximum tumor area found in {orientation.upper()} slice: {max_area_index}")

    return max_area_index


if __name__ == "__main__":
    # Paths to the files
    data_dir = 'sample_data/BraTS-GLI-00000-000'
    flair_file = os.path.join(data_dir, 'BraTS-GLI-00000-000-t2f.nii.gz')
    segmentation_file = os.path.join(data_dir, 'BraTS-GLI-00000-000-seg.nii.gz')
    
    # Load images
    # flair_sitk, flair_np = load_nifti(flair_file)
    # seg_sitk, seg_np = load_nifti(segmentation_file)
    flair_np, affine = load_nifti_with_orientation(flair_file)
    seg_np, _ = load_nifti_with_orientation(segmentation_file)
    
    # Check if sizes match
    if flair_np.shape != seg_np.shape:
        raise ValueError("The sizes of the FLAIR image and the segmentation mask do not match.")
    
    print(f"FLAIR shape: {flair_np.shape}, Segmentation shape: {seg_np.shape}")
    
    # Binarize the mask
    binary_mask = BinarizeMask(seg_np)

    # Extract radiomics features
    bin_width = 15  # Example bin width
    features_3d = radiomics_extractor(flair_np, binary_mask, bin_width)
    
    # Print extracted features
    for key, value in features_3d.items():
        print(f"{key}: {value}")

    #2D Radiomics
    max_area_index = find_index_slice(binary_mask, plot=True)
    flair_slice = flair_np[:, :, max_area_index]
    mask_slice = binary_mask[:, :, max_area_index]

    features_2d = radiomics_extractor(flair_slice, mask_slice, bin_width)
    
    # Print extracted features
    for key, value in features_2d.items():
        print(f"{key}: {value}")

