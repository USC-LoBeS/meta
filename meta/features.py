import os
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
import nibabel as nib
from dipy.tracking.streamline import length
from dipy.io.streamline import load_tractogram


def bundle_length(streamlines):
    return np.mean([length(sl) for sl in streamlines])

def get_span(streamlines):
    spans = [np.linalg.norm(line[0]-line[-1]) for line in streamlines]
    return np.mean(spans)

def get_curl(streamlines):
    return (bundle_length(streamlines) / get_span(streamlines))

def calculate_volume(img):
    data = img.get_fdata()
    voxel_dimensions = img.header.get_zooms()
    voxel_volume = np.prod(voxel_dimensions[:3]) or 1
    return (data > 0).sum() * voxel_volume

def calculate_surface_area(img):
    data = img.get_fdata()
    voxel_dimensions = img.header.get_zooms()
    indices = np.where(data == 1.0)
    neighborhood = list(itertools.product([-1, 0, 1], repeat=3))
    padded_data = np.pad(data, pad_width=1)
    surface_voxels = sum(
        any(padded_data[x + dx + 1, y + dy + 1, z + dz + 1] == 0 for dx, dy, dz in neighborhood)
        for x, y, z in zip(*indices))  
    voxel_area = np.prod(voxel_dimensions[:2]) or 1
    return surface_voxels * voxel_area

def get_diameter(streamlines, img):
    return 2*np.sqrt(calculate_volume(img)/(np.pi*bundle_length(streamlines)))

def get_elongation(streamlines, img):
    return bundle_length(streamlines) /get_diameter(streamlines, img)

def get_irregularity(streamlines, img):
    return calculate_surface_area(img) / (np.pi * (get_diameter(streamlines, img) ** bundle_length(streamlines)))


## Compute bundle features based on binary masks and microstructure maps
def segment_features():
    parser = argparse.ArgumentParser(description='MeTA Feature Calculation based on bundle mask using Pyradiomics')
    parser.add_argument('--subject', type=str, help='subject ID', required=True)
    parser.add_argument('--bundle', type=str, help='bundle name', required=True)
    parser.add_argument('--mask', type=str, help='Paths to mask for ROI', required=True)
    parser.add_argument('--map', type=str, help='Brain microstructure map, e.g. FA, MD, etc.', required=True)
    parser.add_argument('--output', type=str, help='Output path to save extracted features', required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    try:
        features = dict()
        features['subjectID'] = []
        features['bundle'] = []
        features['segment'] = []
        features['mean'] = []

        ## Dictionary to store voxel-wise values
        voxel_features = dict()
        voxel_features['subjectID'] = []
        voxel_features['bundle'] = []
        voxel_features['segment'] = []
        voxel_features['value'] = []

        # Load the mask and microstructure map
        micro_map = nib.load(args.map).get_fdata()
        data = nib.load(args.mask).get_fdata()

        labels = np.unique(data)
        for label in labels:
            label = int(label)
            if label == 0:
                continue

            ## Compute mean for a label of microstructure map
            mean_value = np.nanmean(micro_map[data == label])
            features['subjectID'].append(args.subject)
            features['bundle'].append(args.bundle)
            features['segment'].append(label)
            features['mean'].append(mean_value)

            ## Add the voxel-wise features to the dictionary
            voxel_features['subjectID'].extend([args.subject]*np.sum(data == label))
            voxel_features['bundle'].extend([args.bundle]*np.sum(data == label))
            voxel_features['segment'].extend([label]*np.sum(data == label))
            voxel_features['value'].extend(micro_map[data == label])

        ## Save the features to a csv file
        features_df = pd.DataFrame(features)
        features_df.to_csv(f"{args.output}/{args.subject}_{args.bundle}_segments_average.csv", index=False)

        ## Save the voxel-wise features to a HDF5 file
        voxel_features_df = pd.DataFrame(voxel_features)
        voxel_features_df.to_csv(f"{args.output}/{args.subject}_{args.bundle}_segments_voxelwise.csv", index=False)
        # voxel_features_df.to_hdf(f"{args.output}/{args.subject}_{args.bundle}_segments_voxelwise.h5", key='df', mode='w', complevel=9)
        print('Finished extracting the features')

    except Exception as e:
        print(f'Error in extracting features from {args.mask}: {str(e)}', flush=True)


## Compute bundle features based on streamlines
def streamlines_features():
    parser = argparse.ArgumentParser(description='MeTA Feature Calculation based on bundle streamlines')
    parser.add_argument('--subject', type=str, required=True, help='subject ID')
    parser.add_argument('--bundle', type=str, help='bundle name', required=True)
    parser.add_argument('--mask', type=str, required=True, help='Binary image of the whole bundle')
    parser.add_argument('--tractogram', type=str, required=True, help='streamline file of the bundle')
    parser.add_argument('--output', type=str, required=True, help='Output csv file to save features')

    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    mask = nib.load(args.mask)
    tractogram = load_tractogram(args.tractogram, reference="same", bbox_valid_check=False)

    streamlines_count = len(tractogram.streamlines)
    points = len(tractogram.streamlines.get_data())
    streamlines_length = bundle_length(tractogram.streamlines)
    span = get_span(tractogram.streamlines)
    curl = get_curl(tractogram.streamlines)
    volume = calculate_volume(mask)
    surface_area = calculate_surface_area(mask)
    diameter = get_diameter(tractogram.streamlines, mask)
    elongation = get_elongation(tractogram.streamlines, mask)
    irregularity = get_irregularity(tractogram.streamlines, mask)

    data = {
        'subjectID': args.subject,
        'bundle': args.bundle,
        'streamlines_count': streamlines_count,
        'streamlines_points': points,
        'length': streamlines_length,
        'span': span,
        'curl': curl,
        'volume': volume,
        'surface_area': surface_area,
        'diameter': diameter,
        'elongation': elongation,
        'irregularity': irregularity
    }

    features = pd.DataFrame(data, index=[0])

    if args.output.endswith('.csv'):
        features.to_csv(args.output, index=False)
    else:
        features.to_csv(f"{args.output}/{args.subject}_{args.bundle}_streamlines_metrics.csv", index=False)

    print('Finished extracting the features')
