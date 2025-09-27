import sys
import logging
import numpy as np
import pyvista as pv
import nibabel as nib
from scipy import ndimage
from scipy.spatial.distance import euclidean
from nibabel.affines import apply_affine


logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8',
    level=logging.INFO,
    force=True
)



def process_mesh(mesh,fill_holes=True,fill_size=2500,getLargest=True):
    """
    Cleans and processes a mesh by filling small holes and extracting the largest connected component.

    Parameters:
        mesh: Input mesh object to process.
        fill_holes: If True, fill holes in the mesh up to the specified size.
        fill_size: Maximum size (in area) of holes to be filled.
        getLargest: If True, extract only the largest connected component of the mesh.
    """

    mesh.clean(inplace=True)
    if fill_holes:
        mesh.fill_holes(fill_size,inplace=True)
    if getLargest:
        mesh.extract_largest(inplace=True)
    return mesh.clean(lines_to_points=False, polys_to_lines=False, strips_to_polys=False)


def medial_core(bundle_mask, medial_surface, volume_mesh, percent=0.125, fill=True, size=2500, extract=False):
    """
    Computes the medial core of a white matter bundle based on the medial surface and volume mesh.

    Parameters:
        bundle_mask: Path to the bundle mask in NIfTI format.
        medial_surface: Path to themedial surface mesh in VTK format.
        volume_mesh: Path to the volume mesh in VTK format.
        percent: Percentage of distance to keep from each side of the medial surface.
        fill: If True, fills holes in the bundle mesh.
        size: Maximum size of holes to fill.
        extract: If True, extracts the largest connected component of the bundle mesh.

    Returns:
        core: NIfTI image of the medial core.
        average_depth: Average depth of the medial surface.
    """

    # Read medial surface and volume mesh
    surface = pv.PolyData(medial_surface)
    surface = process_mesh(surface, fill_holes=fill, fill_size=size, getLargest=extract)
    volume = pv.PolyData(volume_mesh)
    volume = process_mesh(volume, fill_holes=fill, fill_size=size, getLargest=extract)

    # Load bundle mask:
    bundle = nib.load(bundle_mask)
    affine = bundle.affine
    mask_data = bundle.get_fdata()
    core_mask = np.zeros(bundle.shape)

    # Compute normals and distances on the medial surface:
    surface_normals = surface.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    surface_normals["distances"] = np.empty(surface_normals.n_points)

    percent_points = []
    for i in range(surface_normals.n_points):
        point = surface_normals.points[i]
        vector = surface_normals["Normals"][i] * surface_normals.length
        point_1 = point - vector
        point_2 = point + vector
        intersection_points, _ = volume.ray_trace(point_1, point_2, first_point=False)
        if len(list(intersection_points))>0:
            distance = np.sqrt(np.sum((intersection_points[0] - point) ** 2))
            surface_normals["distances"][i] = distance
            if intersection_points.shape[0]>2:
                distance_pairs = sorted([(euclidean(point, intersection_point), intersection_point) for intersection_point in list(intersection_points)],key=lambda x:x[0])
                
                percent_points.append(percent * distance_pairs[0][1] + (1 - percent) * point)
                percent_points.append(percent * distance_pairs[1][1] + (1 - percent) * point)

            elif intersection_points.shape[0] == 2:
                percent_points.append(percent * list(intersection_points)[0] + (1 - percent) * point)
                percent_points.append(percent * list(intersection_points)[1] + (1 - percent) * point)

    mask = surface_normals["distances"] == 0
    surface_normals["distances"][mask] = np.nan
    average_depth = np.nanmean(surface_normals["distances"])
    logging.info(f"Average depth: {average_depth}")

    core_points = np.vstack([percent_points, surface.points])
    voxel_indices = np.round(apply_affine(np.linalg.inv(affine), core_points)).astype(int)

    for i in range(voxel_indices.shape[0]):
        if (voxel_indices[i] < core_mask.shape).all() and (voxel_indices[i] >= 0).all():
            core_mask[tuple(voxel_indices[i])] = 1


    voxel_size = np.mean(bundle.header.get_zooms()[:3])
    if voxel_size < 1:
        core_mask_padded = np.pad(core_mask, pad_width=3, mode='constant', constant_values=0)
        core_mask_closed_padded = ndimage.binary_closing(core_mask_padded, structure=np.ones((5,5,5)))
        core_closed = core_mask_closed_padded[3:-3, 3:-3, 3:-3]
    else:
        core_mask_padded = np.pad(core_mask, pad_width=1, mode='constant', constant_values=0)
        core_mask_closed_padded = ndimage.binary_closing(core_mask_padded, structure=np.ones((2,2,2)))
        core_closed = core_mask_closed_padded[1:-1, 1:-1, 1:-1]

    labels, num_labels = ndimage.label(core_closed.astype(np.uint8))
    sizes = ndimage.sum(core_closed, labels, range(num_labels + 1))
    largest_label = np.argmax(sizes[1:]) + 1
    largest_component_mask = (labels == largest_label)
    largest_component_mask *= (mask_data > 0)

    core = nib.Nifti1Image(largest_component_mask.astype(np.uint8), affine) 

    return core, average_depth

