import os
import sys
import logging
import argparse
import numpy as np
import nibabel as nib
from meta.core.mesh import medial_core
from meta.core.segments import segment_bundle
from meta.core.corresponds import get_alignment

import warnings
warnings.simplefilter("ignore")

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8',
    level=logging.INFO,
    force=True
)


def run_main():
    parser = argparse.ArgumentParser(description='Medial Tractography Analysis (MeTA) for White Matter Bundle Parcellation')
    parser.add_argument("--subject", type=str, help='Subject IDs')
    parser.add_argument("--bundle", type=str, help='Name of white matter bundle')
    parser.add_argument("--medial_surface", type=str, help='Medial surface of white matter bundle in vtk format', required=True)
    parser.add_argument("--volume", type=str, help='Volume of white matter bundle in vtk format (mesh)', required=True)
    parser.add_argument("--sbundle", type=str, help='Streamlines of subject bundle', required=True)
    parser.add_argument("--mbundle", type=str, help='Streamlines of model bundle', required=True)
    parser.add_argument("--transform", type=str, help='Transformation matrix: MNI → subject space', required=False, default=None)
    parser.add_argument("--inverse", action='store_true', help='Inverse transformation matrix if direction subject → MNI')
    parser.add_argument("--mask", type=str, help='Mask of white matter bundle', required=True)
    parser.add_argument("--num_segments", type=int, help='The required number of segments along the bundle length', default=15)
    parser.add_argument("--output", type=str, help='Output directory' , required=True)
    parser.add_argument("--percent", type=float, help='Percent of distance to keep from each side of Medial surface' , default=0.125)
    parser.add_argument("--fill", type=bool, help='Fill holes in bundle mesh', default=True)
    parser.add_argument("--size", type=int, help='Maximum hole size to fill', default=2500)
    parser.add_argument("--extract", type=bool, help='Extract largest connected set in bundle mesh', default=False)

    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Extract medial core volume:
    logging.info(f'Computing global core for {args.bundle} bundle ...')
    core, _ = medial_core(bundle_mask = args.mask, medial_surface = args.medial_surface, volume_mesh =args.volume, percent = args.percent, fill = args.fill, size = args.size, extract = args.extract)
    nib.save(core, os.path.join(args.output, args.subject + '_' + args.bundle + "_global_core.nii.gz"))

    logging.info(f'Computing global difference for {args.bundle} bundle ...')

    # Load mask (global all bundle):
    img = nib.load(args.mask)
    mask_data = img.get_fdata()

    core_data = core.get_fdata()
    core_indices = np.where(core_data > 0)
    core_data[core_indices]=mask_data[core_indices]
    global_diff = mask_data-core_data
    diff_indices = np.where(global_diff > 0)
    global_diff[diff_indices]=mask_data[diff_indices]
    nib.save(nib.Nifti1Image(global_diff, affine=img.affine), filename=os.path.join(args.output, args.subject + '_' + args.bundle + "_global_diff.nii.gz"))

    ## Get corresponding points between model and subject bundle:
    logging.info(f'Getting corresponding points between model and subject bundle...')
    corres_points, original_indices, n_segments = get_alignment(model = args.mbundle, subject = args.sbundle, n_segments = args.num_segments, mask_img = args.mask, transform = args.transform, inverse = args.inverse)

    ## Segment the bundle using DTW points:
    logging.info(f'Segmenting {args.bundle} into {n_segments} segments')
    segments = segment_bundle(bundle_data = mask_data, dtw_points_sets = corres_points, num_segments = n_segments)

    segmented_bundle = np.zeros(mask_data.shape)
    labels = original_indices + 1
    for seg_bool, label in zip(segments, labels):
        segmented_bundle[seg_bool] = label
    core_bundle = core.get_fdata()
    core_indices = np.where(core_bundle > 0)
    core_bundle[core_indices] = segmented_bundle[core_indices]
    diff_bundle = segmented_bundle - core_bundle
    diff_indices = np.where(diff_bundle > 0)
    diff_bundle[diff_indices] = segmented_bundle[diff_indices]

    logging.info('Saving segmentation along length results...')
    # Local all segments:
    local_all = nib.Nifti1Image(segmented_bundle, affine=img.affine)
    nib.save(local_all, os.path.join(args.output, args.subject + '_' + args.bundle + '_' + str(args.num_segments) + "_segments_local_all.nii.gz"))

    # Local core segments:
    local_core = nib.Nifti1Image(core_bundle, affine=img.affine)
    nib.save(local_core, os.path.join(args.output, args.subject + '_' + args.bundle + '_' + str(args.num_segments) + "_segments_local_core.nii.gz"))

    # Local difference segments:
    local_diff = nib.Nifti1Image(diff_bundle, affine=img.affine)
    nib.save(local_diff, os.path.join(args.output, args.subject + '_' + args.bundle + '_' + str(args.num_segments) + "_segments_local_diff.nii.gz"))

if __name__ == '__main__':
    run_main()

