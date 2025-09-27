import sys
import logging
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import map_coordinates
from meta.io.streamline import read_streamlines
from dipy.tracking.streamline import transform_streamlines


logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8',
    level=logging.INFO,
    force=True
)

## Compute bundle features based on binary masks and microstructure maps
def volumetric_profile():
    parser = argparse.ArgumentParser(description='Compute volumetric profile (average mean and voxel-wise) of a white matter bundle.')
    parser.add_argument('--subject', type=str, help='Subject ID', required=True)
    parser.add_argument('--bundle', type=str, help='White matter bundle name', required=True)
    parser.add_argument('--mask', type=str, help='Path to white matter bundle mask', required=True)
    parser.add_argument('--map', type=str, help='Brain microstructure map, e.g. FA, MD, etc.', required=True)
    parser.add_argument('--output', type=str, help='Output directory to save extracted features', required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    try:
        features = {
            'subjectID': [],
            'bundle': [],
            'segment': [],
            'mean': []
        }
        
        voxel_features = {
            'subjectID': [],
            'bundle': [],
            'segment': [],
            'value': []
        }

        # Load the mask and microstructure map
        micro_map = nib.load(args.map).get_fdata()
        data = nib.load(args.mask).get_fdata()

        labels = np.unique(data)
        for label in labels:
            label = int(label)
            if label == 0:
                continue

            ## Compute mean for a label of microstructure map
            mean_value = np.nanmean(micro_map[(data == label) & (micro_map != 0)])
            features['subjectID'].append(args.subject)
            features['bundle'].append(args.bundle)
            features['segment'].append(label)
            features['mean'].append(mean_value)

            ## Add the voxel-wise values of the microstructure map:
            voxel_features['subjectID'].extend([args.subject]*np.sum(data == label))
            voxel_features['bundle'].extend([args.bundle]*np.sum(data == label))
            voxel_features['segment'].extend([label]*np.sum(data == label))
            voxel_features['value'].extend(micro_map[data == label])

        mean_df = pd.DataFrame(features)
        csv_path = f"{args.output}/{args.subject}_{args.bundle}_segments_average.csv"
        mean_df.to_csv(csv_path, index=False)
        logging.info(f"Saved segment-wise mean features to {csv_path}")

        ## Save the voxel-wise features to HDF5:
        voxel_wise_df = pd.DataFrame(voxel_features)
        # voxel_wise_df.to_csv(f"{args.output}/{args.subject}_{args.bundle}_segments_voxelwise.csv", index=False)
        hdf5_path = f"{args.output}/{args.subject}_{args.bundle}_segments_voxelwise.h5"
        voxel_wise_df.to_hdf(hdf5_path, key='df', mode='w', complevel=9)
        logging.info(f"Saved voxel-wise features to {hdf5_path}")

    except Exception as e:
        logging.error(f'An error occurred while processing {args.mask}: {str(e)}')


def streamlines_profile():
    parser = argparse.ArgumentParser(description='Compute streamlines profile (average mean and point-wise) of a white matter bundle.')
    parser.add_argument("--subject", type=str, help='Subject ID', required=True)
    parser.add_argument("--bundle", type=str, help='White matter bundle name', required=True)
    parser.add_argument("--sbundle", type=str, help='Streamlines of subject bundle', required=True)
    parser.add_argument("--mask", type=str, help='Path to white matter bundle mask', required=True)
    parser.add_argument("--map", type=str, help='Brain microstructure map, e.g. FA, MD, etc.', required=True)
    parser.add_argument("--output", type=str, help='Output directory to save extracted features', required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    try:
        ## metrics map and bundle mask/labels:
        micro_map = nib.load(args.map)
        metric_data = micro_map.get_fdata()

        labels_img = nib.load(args.mask)
        labels_data = labels_img.get_fdata()

        ## Load streamlines:
        streamlines = read_streamlines(args.sbundle)
        print(f"Number of streamlines: {len(streamlines)}")
        sls_voxels = transform_streamlines(streamlines, np.linalg.inv(micro_map.affine))
        sls_pts = np.vstack(sls_voxels)

        ## Extract microstructure values and labels for each point along the streamlines:
        metric_vals = map_coordinates(metric_data, sls_pts.T, order=0, mode="nearest")
        label_vals  = map_coordinates(labels_data, sls_pts.T, order=0, mode="nearest").astype(int)

        ## Save the point-wise features to HDF5:
        df = pd.DataFrame({
            "subjectID" : args.subject,
            "bundle" : args.bundle,
            "segment" : label_vals,
            "value": metric_vals
        })
        df = df[df.segment != 0]
        print(df.segment.unique())

        hdf5_path = f"{args.output}/{args.subject}_{args.bundle}_streamlines_points.h5"
        df.to_hdf(hdf5_path, key='df', mode='w', complevel=9)
        logging.info(f"Saved streamlines point-wise features to {hdf5_path}")

        ## Compute mean for each segment:
        df_mean = (
            df.groupby(["subjectID", "bundle", "segment"], as_index=False)
            .agg(mean=("value", "mean"))
        )
        csv_path = f"{args.output}/{args.subject}_{args.bundle}_streamlines_mean.csv"
        df_mean.to_csv(csv_path, index=False)
        logging.info(f"Saved streamlines segment-wise mean features to {csv_path}")

    except Exception as e:
        logging.error(f'An error occurred while processing {args.mask}: {str(e)}')

