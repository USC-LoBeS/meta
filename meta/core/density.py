import sys
import logging
import argparse
import nibabel as nib
from dipy.tracking import utils
from meta.io.streamline import read_streamlines


logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8',
    level=logging.INFO,
    force=True
)


def density_map():

    parser = argparse.ArgumentParser(description='Convert streamlines of white matter bundle into a density map and binary mask.')
    parser.add_argument('--bundle', type=str, help='Path to the bundle file containing streamlines', required=True)
    parser.add_argument('--reference', type=str, help='Path to the reference image', required=True)
    parser.add_argument('--output', type=str, help='Path to the output binary mask file', required=True)
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    # Load reference image
    ref_img = nib.load(args.reference)
    ref_affine = ref_img.affine
    ref_shape = ref_img.get_fdata().shape

    # Load streamlines
    streamlines = read_streamlines(args.bundle)
    max_seq_len = abs(ref_affine[0, 0] / 4)
    streamlines = list(utils.subsegment(streamlines, max_seq_len))

    # Create Density Map
    density_map = utils.density_map(streamlines, vol_dims=ref_shape, affine=ref_affine)

    ## Convert Density Map to Binary Mask
    mask = density_map > 0
    nib.save(nib.Nifti1Image(mask.astype("uint8"), ref_affine), args.output)
    logging.info(f'Binary mask saved to {args.output}')

    ## save the density map
    density_img = nib.Nifti1Image(density_map.astype("float32"), ref_affine)
    nib.save(density_img, args.output.replace('.nii.gz', '_density.nii.gz'))
    logging.info(f'Density map and binary mask saved to {args.output.replace(".nii.gz", "_density.nii.gz")}')
