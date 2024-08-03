import os
import sys 
import argparse
import numpy as np
import pyvista as pv
import nibabel as nib

from meta.utils import processMesh, compute_core
from meta.bundle_segmentation import segment_bundle, perform_dtw

import warnings
warnings.simplefilter("ignore")

def run_main():
    parser = argparse.ArgumentParser(description='Medial Tractography Analysis (MeTA) for White Matter Bundle Parcellation')
    parser.add_argument('--subject', type=str, help='Subject IDs')
    parser.add_argument('--bundle', type=str, help='Name of white matter bundle')
    parser.add_argument("--medial_surface", type=str, help='Medial surface of white matter bundle in vtk format', required=True)
    parser.add_argument("--volume", type=str, help='Volume of white matter bundle in vtk format (mesh)', required=True)
    parser.add_argument("--sbundle", type=str, help='Streamlines of subject bundle', required=True)
    parser.add_argument("--mbundle", type=str, help='Streamlines of model bundle', required=True)
    parser.add_argument("--transform", type=str, help='Transformation matrix to subject space', required=False, default=None)
    parser.add_argument("--mask", type=str, help='Mask of white matter bundle', required=True)
    parser.add_argument("--num_segments", type=int, help='Number of segments', default=15)
    parser.add_argument("--output", type=str, help='Output directory' , required=True)
    parser.add_argument("--percent", type=float, help='Percent of distance to keep from each side of Medial surface' , default=0.125)
    parser.add_argument("--fill", type=bool, help='Fill holes in bundle mesh', default=True)
    parser.add_argument("--size", type=int, help='Maximum hole size to fill', default=2500)
    parser.add_argument("--extract", type=bool, help='Extract largest connected set in bundle mesh', default=False)

    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    cmrep_surface = pv.PolyData(args.medial_surface) 
    cmrep_surface = processMesh(cmrep_surface,fillHoles=args.fill, fillSize=args.size, getLargest=args.extract)
    bundle_volume = pv.PolyData(args.volume) 
    bundle_volume = processMesh(bundle_volume,fillHoles=args.fill, fillSize=args.size, getLargest=args.extract)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    img = nib.load(args.mask)
    mask_data = img.get_fdata()
    print(f'Computing global core for {args.bundle} bundle ...', flush=True)
    core, average_depth = compute_core(args.mask, cmrep_surface, bundle_volume, percent=args.percent)
    nib.save(core, os.path.join(args.output, args.subject + '_' + args.bundle + "_global_core.nii.gz"))

    print(f'Computing global difference for {args.bundle} bundle ...', flush=True)
    core_data = core.get_fdata()
    core_indices = np.where(core_data > 0)
    core_data[core_indices]=mask_data[core_indices]
    global_diff = mask_data-core_data
    diff_indices = np.where(global_diff > 0)
    global_diff[diff_indices]=mask_data[diff_indices]
    nib.save(nib.Nifti1Image(global_diff, affine=img.affine), filename=os.path.join(args.output, args.subject + '_' + args.bundle + "_global_diff.nii.gz"))

    print('Getting corresponding points between model and subject bundle...', flush=True)
    dtw_points_sets = perform_dtw(args.mbundle, args.sbundle, args.num_segments, mask_img=args.mask, transform=args.transform)

    print(f'Segmenting {args.bundle} into {args.num_segments} segments', flush=True)
    segments = segment_bundle(mask_data, dtw_points_sets, args.num_segments) 
    segmented_bundle=np.zeros(mask_data.shape)
    numIntensities = len(segments)
    intensities = [i+1 for i in range(numIntensities)]
    for i,j in zip(segments,intensities):
        segmented_bundle+=((i)*j)
    core_bundle = core.get_fdata()
    core_indices = np.where(core_bundle > 0)
    core_bundle[core_indices]=segmented_bundle[core_indices]
    diff_bundle = segmented_bundle-core_bundle
    diff_indices = np.where(diff_bundle > 0)
    diff_bundle[diff_indices]=segmented_bundle[diff_indices]

    print('Saving segmentation ...', flush=True)
    nib.save(nib.Nifti1Image(segmented_bundle, affine=img.affine), filename=os.path.join(args.output, args.subject + '_' + args.bundle + '_' + str(args.num_segments) + "_segments_local_all.nii.gz"))
    nib.save(nib.Nifti1Image(core_bundle, affine=img.affine), filename=os.path.join(args.output, args.subject + '_' + args.bundle + '_' + str(args.num_segments) + "_segments_local_core.nii.gz"))
    nib.save(nib.Nifti1Image(diff_bundle, affine=img.affine), filename=os.path.join(args.output, args.subject + '_' + args.bundle + '_' + str(args.num_segments) + "_segments_local_diff.nii.gz"))

if __name__ == '__main__':
    run_main()
