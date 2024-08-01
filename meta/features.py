import os
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
import nibabel as nib
from radiomics import featureextractor
from dipy.tracking.streamline import length
from dipy.io.streamline import load_tractogram


def calculate_dice(core, all):
    core_mask = core.get_fdata().astype(bool)
    all_mask = all.get_fdata().astype(bool)
    assert core_mask.shape==all_mask.shape, "Size mismatch, core and all should have the same size"
    intersection = np.count_nonzero(core_mask & all_mask)
    dice_coeff = 2. * intersection / (np.count_nonzero(core_mask) + np.count_nonzero(all_mask))
    fractional_volume = intersection / np.count_nonzero(all_mask)
    return dice_coeff, fractional_volume

def get_length(streamlines):
    return np.mean(list(length(streamlines)))

def get_span(streamlines):
    spans = [np.linalg.norm(line[0]-line[-1]) for line in streamlines]
    return np.mean(spans)

def get_curl(streamlines):
    return (get_length(streamlines) / get_span(streamlines))

def calculate_volume(img):
    data = img.get_fdata()
    voxel_dimensions = img.header.get_zooms()
    voxel_volume = np.prod(voxel_dimensions[:3]) or 1
    return (data == 1.0).sum() * voxel_volume

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
    return 2*np.sqrt(calculate_volume(img)/(np.pi*get_length(streamlines)))

def get_elongation(streamlines, img):
    return get_length(streamlines) /get_diameter(streamlines, img)

def get_irregularity(streamlines, img):
    return calculate_surface_area(img) / (np.pi * (get_diameter(streamlines, img) ** get_length(streamlines)))


def segment_features():
    parser = argparse.ArgumentParser(description='MeTA Feature Calculation based on bundle mask using Pyradiomics')
    parser.add_argument('--subject', type=str, help='subject ID', required=True)
    parser.add_argument('--bundle', type=str, help='bundle name', required=True)
    parser.add_argument('--mask', type=str, nargs='+', help='list of paths to ROIs', required=True)
    parser.add_argument('--map', type=str, help='Brain microstructure map, e.g. FA, MD, etc.', required=True)
    parser.add_argument('--params', type=str, help='Pyradiomics parameters config. file in yaml format')
    parser.add_argument('--output', type=str, help='Output csv file to save features', required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    if args.params is None:
        params = os.path.dirname(__file__) + '/params.yaml'
        print('Using default parameters: ', params, flush=True)
    extractor = featureextractor.RadiomicsFeatureExtractor(params)

    features = dict()
    features['subjectID'] = []
    features['region'] = []
    features['segment'] = []

    for mask in args.mask:
        try:
            data = nib.load(mask).get_fdata()
            labels = np.unique(data)
            for label in labels:
                label = int(label)
                if label == 0:
                    continue

                Pyradiomics = extractor.execute(imageFilepath=args.map, maskFilepath=mask, label=label)
                features['subjectID'].append(args.subject)
                features['region'].append(args.bundle)
                features['segment'].append(label)

                for item in Pyradiomics.keys():
                    try:
                        value = float(Pyradiomics[item])
                        if item not in features.keys():
                            features[item] = []
                        features[item].append(value)
                    except Exception:
                        pass
        except Exception:
            print('Error in extracting features from ', mask, flush=True)
            print('Mask may only contains 1 segmented voxel! Cannot extract features for a single voxel', flush=True) 

    features = pd.DataFrame(features)
    if args.output.endswith('.csv'):
        features.to_csv(args.output, index=False)
    else:
        features.to_csv(f"{args.output}/{args.subject}_{args.bundle}_segment_metrics.csv", index=False)
    
    print('Finished extracting the features')


def streamlines_features():
    parser = argparse.ArgumentParser(description='MeTA Feature Calculation based on bundle streamlines')
    parser.add_argument('--subject', type=str, required=True, help='subject ID')
    parser.add_argument('--bundle', type=str, help='bundle name', required=True)
    parser.add_argument('--core', type=str, required=True, help='Global binary image of the bundle core')
    parser.add_argument('--all', type=str, required=True, help='Global binary image of the whole bundle')
    parser.add_argument('--tractogram', type=str, required=True, help='streamline file of the bundle')
    parser.add_argument('--output', type=str, required=True, help='Output csv file to save features')

    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()


    core = nib.load(args.core)
    all = nib.load(args.all)
    tractogram = load_tractogram(args.tractogram, reference="same", bbox_valid_check=False)

    dice_coeff, fractional_volume = calculate_dice(core, all)
    streamlines_count = len(tractogram.streamlines)
    points = len(tractogram.streamlines.get_data())
    length = get_length(tractogram.streamlines)
    span = get_span(tractogram.streamlines)
    curl = get_curl(tractogram.streamlines)
    volume = calculate_volume(all)
    surface_area = calculate_surface_area(all)
    diameter = get_diameter(tractogram.streamlines, all)
    elongation = get_elongation(tractogram.streamlines, all)
    irregularity = get_irregularity(tractogram.streamlines, all)

    data = {
        'subjectID': args.subject,
        'bundle': args.bundle,
        'dice_coefficient': dice_coeff,
        'fractional_volume': fractional_volume,
        'streamlines_count': streamlines_count,
        'streamlines_points': points,
        'length': length,
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