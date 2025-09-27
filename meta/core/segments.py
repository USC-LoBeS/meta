import numpy as np
from tqdm import tqdm


def segment_bundle(bundle_data, dtw_points_sets, num_segments):
    """
    Parcellate white matter bundle into specified segments based on DTW points.

    Parameters:
    -----------
    bundle_data: A binary mask of the white matter bundle as a NumPy array
    dtw_points_sets: A list of arrays with shape (num_segments, 3) which are the corresponding DTW points.
    num_segments: The required number of segments to divide the bundle into

    Returns:
    --------
    segments: A list of labels, where each label corresponds to a segment.
    """

    segments = [np.zeros_like(bundle_data, dtype=bool) for _ in range(num_segments+1)]

    for dtw_points in tqdm(dtw_points_sets):
        for i in range(num_segments):

            if i == 0:
                plane_normal = (dtw_points[i+1] - dtw_points[i]).astype(float)
                for x, y, z in np.argwhere(bundle_data):
                    point = np.array([x, y, z])
                    if np.dot(point - dtw_points[i], -plane_normal) >= 0:
                        segments[i][x, y, z] = True

            if i < num_segments - 2 and i >= 0:
                plane_normal = (dtw_points[i+1] - dtw_points[i]).astype(float)
                next_plane_normal = (dtw_points[i+1 + 1] - dtw_points[i+1]).astype(float)
                for x, y, z in np.argwhere(bundle_data):
                    point = np.array([x, y, z])
                    if np.dot(point - dtw_points[i], plane_normal) >= 0 and np.dot(point - dtw_points[i+1], -next_plane_normal) >= 0:
                        segments[i+1][x, y, z] = True

            elif i == num_segments - 2: 
                plane_normal = (dtw_points[i] - dtw_points[i-1]).astype(float)
                for x, y, z in np.argwhere(bundle_data):
                    point = np.array([x, y, z])
                    if np.dot(point - dtw_points[i-1], plane_normal) >= 0:
                        segments[i+1][x, y, z] = True

            elif i == num_segments - 1:
                plane_normal = (dtw_points[i] - dtw_points[i-1]).astype(float)
                for x, y, z in np.argwhere(bundle_data):
                    point = np.array([x, y, z])
                    if np.dot(point - dtw_points[i], plane_normal) >= 0:
                        segments[i+1][x, y, z] = True

    arrays = np.array(segments)
    sum_array = np.sum(arrays, axis=0)
    remaining_voxels = sum_array.copy()
    if np.any(remaining_voxels):
        for x, y, z in np.argwhere(sum_array >= 2):
            for seg in segments:
                seg[x, y, z] = False
            point = np.array([x, y, z])
            min_distance = float('inf')
            closest_segment_idx = None
            for dtw_points in dtw_points_sets:
                for i in range(num_segments):
                    distance_to_start = np.linalg.norm(point - dtw_points[i])
                    if distance_to_start < min_distance:
                        min_distance = distance_to_start
                        closest_segment_idx = i
            if closest_segment_idx is not None:
                segments[closest_segment_idx][x, y, z] = True    
    return segments


