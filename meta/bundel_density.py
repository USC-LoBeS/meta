import sys
import argparse
import nibabel as nib
from dipy.tracking import utils as utils_trk

def bundle_density():

    parser = argparse.ArgumentParser(description='Convert a trk streamline file to a binary map.')
    parser.add_argument('--bundle', type=str, help='Input streamline file')
    parser.add_argument('--output', type=str, help='Output mask file')
    parser.add_argument('--reference', type=str, help='Reference image')

    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    ref_img = nib.load(args.reference)
    ref_affine = ref_img.affine
    ref_shape = ref_img.get_fdata().shape

    sl_file = nib.streamlines.load(args.bundle)
    streamlines = sl_file.streamlines
    
    # Upsample Streamlines
    max_seq_len = abs(ref_affine[0, 0] / 4)
    streamlines = list(utils_trk.subsegment(streamlines, max_seq_len))
    # Create Density Map
    dm = utils_trk.density_map(streamlines, vol_dims=ref_shape, affine=ref_affine)
    # Create Binary Map
    dm_binary = dm > 0

    dm_binary_img = nib.Nifti1Image(dm_binary.astype("uint8"), ref_affine)
    nib.save(dm_binary_img, args.output)