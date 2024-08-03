import os
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.io import loadmat
from nibabel.affines import apply_affine


def processMesh(mesh,fillHoles=True,fillSize=2500,getLargest=True):
    mesh.clean(inplace=True)
    if fillHoles:
        mesh.fill_holes(fillSize,inplace=True)
    if getLargest:
        mesh.extract_largest(inplace=True)
    return mesh.clean(lines_to_points=False, polys_to_lines=False, strips_to_polys=False)

def compute_distance(a,b):
    """
    Args:
    a,b - np.array 3D vectors
    Returns Euclidean distance between the two vectors
    """
    return np.linalg.norm(a-b)

def compute_core(mask_path, surface, volume, percent=0.125):
    """
    Takes in input volume surface and the medial surface mesh
    Implements Ray tracing logic to compute new points at specified percent depth.
    Also computes average depth from outer volume to medial surface.
    Args:
        mask_path (str): Path to the Nifti mask file
        medial_surface (PolyData): Bundle Medial surface
        volume (PolyData): Bundle mesh volume
        percent (float): specified percent for depth calculation (default is 12.5%)
    Returns:
        core_nii (Nifti1Image): Nifti image of the core mask
    """
    
    # Load the mask
    bundle = nib.load(mask_path)
    affine = bundle.affine
    mask_data = bundle.get_fdata()
    core_mask = np.zeros(bundle.shape)
    
    medial_surface_normals = surface.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    medial_surface_normals["distances"] = np.empty(medial_surface_normals.n_points)
    specific_percent_points, specific_percent_points_prime = [], []
    for i in range(medial_surface_normals.n_points):
        point = medial_surface_normals.points[i]
        vector = medial_surface_normals["Normals"][i] * medial_surface_normals.length
        point0 = point - vector
        point1 = point + vector
        intersection_points, intersection_cells = volume.ray_trace(point0, point1, first_point=False)
        if len(list(intersection_points))>0:
            distance = np.sqrt(np.sum((intersection_points[0] - point) ** 2))
            medial_surface_normals["distances"][i] = distance
            if intersection_points.shape[0]>2:
                distance_pairs = sorted([(compute_distance(point, intersection_point), intersection_point) for intersection_point in list(intersection_points)],key=lambda x:x[0])

                point_at_percent_1 = distance_pairs[0][1]
                scaled_point_1 = percent * point_at_percent_1 + (1 - percent) * point
                specific_percent_points.append(scaled_point_1)

                point_at_percent_2 = distance_pairs[1][1]
                scaled_point_2 = percent * point_at_percent_2 + (1 - percent) * point
                specific_percent_points_prime.append(scaled_point_2)

            elif intersection_points.shape[0] == 2:

                point_at_percent_1 = list(intersection_points)[0]
                scaled_point_1 = percent * point_at_percent_1 + (1 - percent) * point
                specific_percent_points.append(scaled_point_1)

                point_at_percent_2 = list(intersection_points)[1]
                scaled_point_2 = percent * point_at_percent_2 + (1 - percent) * point
                specific_percent_points_prime.append(scaled_point_2)

    mask = medial_surface_normals["distances"] == 0
    medial_surface_normals["distances"][mask] = np.nan
    average_depth = np.nanmean(medial_surface_normals["distances"])
    
    all_points = np.vstack([specific_percent_points, specific_percent_points_prime, surface.points])

    voxel_indices = np.round(apply_affine(np.linalg.inv(affine), all_points)).astype(int)

    for i in range(voxel_indices.shape[0]):
        if (voxel_indices[i] < core_mask.shape).all() and (voxel_indices[i] >= 0).all():
            core_mask[tuple(voxel_indices[i])] = 1

    voxel_size = sum(bundle.header.get_zooms()[:3]) / 3
    if voxel_size < 1:
        core_mask_padded = np.pad(core_mask, pad_width=3, mode='constant', constant_values=0)
        core_mask_closed_padded = ndimage.binary_closing(core_mask_padded, structure=np.ones((5,5,5)))
        core_mask_closed = core_mask_closed_padded[3:-3, 3:-3, 3:-3]
    else:
        core_mask_padded = np.pad(core_mask, pad_width=1, mode='constant', constant_values=0)
        core_mask_closed_padded = ndimage.binary_closing(core_mask_padded, structure=np.ones((2,2,2)))
        core_mask_closed = core_mask_closed_padded[1:-1, 1:-1, 1:-1]


    labels, num_labels = ndimage.label(core_mask_closed.astype(np.uint8))
    sizes = ndimage.sum(core_mask_closed, labels, range(num_labels + 1))
    largest_label = np.argmax(sizes[1:]) + 1
    largest_component_mask = (labels == largest_label)
    largest_component_mask *= (mask_data > 0)

    core = nib.Nifti1Image(largest_component_mask.astype(np.uint8), affine)

    return core, average_depth


## https://github.com/scilus/scilpy/blob/48911befe7049711536c7dd649e385ca30a9bbc0/scilpy/io/utils.py#L802
def load_matrix_in_any_format(filepath):
    _, ext = os.path.splitext(filepath)
    if ext == '.txt':
        data = np.loadtxt(filepath)
    elif ext == '.npy':
        data = np.load(filepath)
    elif ext == '.mat':
        transfo_dict = loadmat(filepath)
        lps2ras = np.diag([-1, -1, 1])
        transfo_key = 'AffineTransform_double_3_3'
        if transfo_key not in transfo_dict:
            transfo_key = 'AffineTransform_float_3_3'

        rot = transfo_dict[transfo_key][0:9].reshape((3, 3))
        trans = transfo_dict[transfo_key][9:12]
        offset = transfo_dict['fixed']
        r_trans = (np.dot(rot, offset) - offset - trans).T * [1, 1, -1]

        data = np.eye(4)
        data[0:3, 3] = r_trans
        data[:3, :3] = np.dot(np.dot(lps2ras, rot), lps2ras)
    else:
        raise ValueError('Extension {} is not supported'.format(ext))
    return data


def reorient_streamlines(m_centroid, s_centroids):
    """
    Reorients the subject centroids based on the model centroid.
    Args:
        m_centroid (np.ndarray): Model centroid
        s_centroids (list): List of subject centroids
    Returns:
        oriented_s_centroids (list): List of reoriented subject centroids
    """

    def is_flipped(m_centroid, s_centroid):
        """
        checks if subjects centroid is flipped compared to the model centroid.
        """
        start_distance = np.linalg.norm(m_centroid[0] - s_centroid[-1])
        start = np.linalg.norm(m_centroid[-1] - s_centroid[0])

        end_distance = np.linalg.norm(m_centroid[-1] - s_centroid[-1])
        end = np.linalg.norm(m_centroid[0] - s_centroid[0])

        if (start_distance < end_distance) and (start < end):
            return True
        return False

    oriented_s_centroids = []
    for s_centroid in s_centroids:
        if is_flipped(m_centroid, s_centroid):
            oriented_s_centroids.append(s_centroid[::-1])
        else:
            oriented_s_centroids.append(s_centroid)

    return oriented_s_centroids
