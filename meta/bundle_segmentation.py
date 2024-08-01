import numpy as np
import nibabel as nib
from tqdm import tqdm
from tslearn.metrics import dtw_path

from meta.utils import reorient_streamlines
from meta.utils import load_matrix_in_any_format

from dipy.io.streamline import load_tractogram
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import length, transform_streamlines


def segment_bundle(bundle_data, dtw_points_sets, num_segments):
    """
    Parcellate white matter bundle into num_segments based on DTW points.

    Parameters:
    -----------
    bundle_data: A bundle mask as a NumPy array.
    dtw_points_sets: list of ndarrays of shape (num_segments, 3) which are the corresponding DTW points.
    num_segments (int): required number of segments.

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

def perform_dtw(model_bundle, subject_bundle, num_segments, mask_img=None, transform=None):
    """
    This function performs Dynamic Time Warping (DTW) on two tractogram (.trk)
    files in same space.

    Args:
        tbundle (str): path to a template .trk file
        sbundle (str): Path to a subject .trk file
        num_segments (int): number of points (N+1) of template centroid to segment the bundle (N)

    Returns:
        dict: dictionary containing the corresponding points.
    """

    reference_image = nib.load(mask_img)

    ## Trasform the Template bundle to the subject space world cordinates and then to the subject voxel space cordinates:
    model_streamlines = load_tractogram(model_bundle, "same", bbox_valid_check=False).streamlines

    if transform is not None:
        transform_matrix = load_matrix_in_any_format(transform)
        transformed_model_bundles = transform_streamlines(model_streamlines, transform_matrix)
        transformed_model_bundles = transform_streamlines(transformed_model_bundles, np.linalg.inv(reference_image.affine))
    else:
        transformed_model_bundles = transform_streamlines(model_streamlines, np.linalg.inv(reference_image.affine))

    m_feature = ResampleFeature(nb_points=num_segments)
    m_metric = AveragePointwiseEuclideanMetric(m_feature)
    m_qb = QuickBundles(threshold=np.inf, metric=m_metric)
    m_centroid = m_qb.cluster(transformed_model_bundles).centroids
    print('Model: Centroid length... ', np.mean([length(streamline) for streamline in m_centroid]))

    ## Trasform the Subject bundle to the subject voxel cordinates:
    subject_streamlines = load_tractogram(subject_bundle, "same", bbox_valid_check=False).streamlines
    transformed_subject_bundles = transform_streamlines(subject_streamlines, np.linalg.inv(reference_image.affine))
    s_feature = ResampleFeature(nb_points=500)
    s_metric = AveragePointwiseEuclideanMetric(s_feature)
    s_qb = QuickBundles(threshold=np.inf, metric=s_metric)
    s_centroid = s_qb.cluster(transformed_subject_bundles).centroids
    print('Subject: Centroid length... ', np.mean([length(streamline) for streamline in s_centroid]))

    ## Create multiple centroids from subject bundle using QuickBundles
    num_clusters = 100
    feature = ResampleFeature(nb_points=500)
    metric = AveragePointwiseEuclideanMetric(feature)
    qb = QuickBundles(threshold=2., metric=metric, max_nb_clusters=num_clusters)
    centroids = qb.cluster(transformed_subject_bundles).centroids

    ## Check if the centroids are flipped compared to the model centroid
    s_centroid = reorient_streamlines(m_centroid, s_centroid)
    centroids = reorient_streamlines(m_centroid, centroids)

    ## Compute the correspondence between the model and the subject centroids using DTW
    dtw_corres = []
    for idx, (m_centroid, s_centroid) in enumerate(zip(m_centroid, s_centroid)):
        pathDTW, similarityScore = dtw_path(m_centroid, s_centroid)
        x1, y1, z1 = m_centroid[:, 0], m_centroid[:, 1], m_centroid[:, 2]
        x2, y2, z2 = s_centroid[:, 0], s_centroid[:, 1], s_centroid[:, 2]
        corres = dict()
        for (i, j) in pathDTW:
            key = (x1[i], y1[i], z1[i])
            value = (x2[j], y2[j], z2[j])
            if key in corres:
                corres[key].append(value)
            else:
                corres[key] = [value]
        centroid_corres = []
        for key in corres.keys():
            t = len(corres[key]) // 2
            centroid_corres.append(corres[key][t])
        dtw_corres.append(np.array(centroid_corres))

    ## Establish correspondence between dtw_corres and centroids of the subject bundle
    s_corres = []
    for idx, centroid in enumerate(centroids):

        s_centroid = np.squeeze(centroid)
        s_ref  = np.squeeze(dtw_corres)
        pathDTW, similarityScore = dtw_path(s_ref, s_centroid)
        x1, y1, z1 = s_ref[:, 0], s_ref[:, 1], s_ref[:, 2]
        x2, y2, z2 = s_centroid[:, 0], s_centroid[:, 1], s_centroid[:, 2]
        corres = dict()
        for (i, j) in pathDTW:
            key = (x1[i], y1[i], z1[i])
            value = (x2[j], y2[j], z2[j])
            if key in corres:
                corres[key].append(value)
            else:
                corres[key] = [value]

        centroid_corres = []
        for key in corres.keys():
            t = len(corres[key]) // 2
            centroid_corres.append(corres[key][t])
        s_corres.append(np.array(centroid_corres))

    ## combine correspondences
    combined_corres = dtw_corres + s_corres

    ## Remove centroids that are shorter than the threshold
    data = []
    for streamline in combined_corres:
        data.append(length(streamline))  
    mean_length = np.mean(data)
    std_length = np.std(data)
    print("Average streamlines length", np.mean(data))
    print("Standard deviation", std_length)
    threshold = mean_length - 1 * std_length
    indices = np.where(data < threshold)
    final_corres = [sl for idx, sl in enumerate(combined_corres) if idx not in indices[0]]

    ## Compute pairwise distances between corresponding points of the final centroids
    corresponding_points = np.array(final_corres)
    pairwise_distances = np.zeros((corresponding_points.shape[1], corresponding_points.shape[0], corresponding_points.shape[0]))
    for i in range(corresponding_points.shape[1]):
        for j in range(corresponding_points.shape[0]):
            for k in range(j + 1, corresponding_points.shape[0]):
                pairwise_distances[i, j, k] = np.linalg.norm(corresponding_points[j, i] - corresponding_points[k, i])
    pairwise_distances[pairwise_distances == 0] = np.nan
    mean_distances = np.nanmean(pairwise_distances, axis=(1, 2))
    std_distances = np.nanstd(pairwise_distances, axis=(1, 2))
    excluded_idx = np.where(std_distances <= 3.5)[0]

    ## Filter the final_corres based on pairwise distances that have std <= 3.5
    excluded_start = excluded_idx[0]
    excluded_end = excluded_idx[-1]

    filtered_arrays = []
    for idx, array in enumerate(final_corres):
        combined_array = []
        if excluded_start > 1:
            start_point = array[0]
            end_point = array[excluded_start]
            side_1_points = np.linspace(start_point, end_point, excluded_start + 1)[1:-1]
            combined_array.extend(array[0:1])
            combined_array.extend(side_1_points)
        elif excluded_start <= 1:
            combined_array.extend(array[0:excluded_start])
        combined_array.extend(array[excluded_start:excluded_end+1])
        if num_segments - excluded_end > 1:
            start_point = array[excluded_end]
            end_point = array[-1]
            side_2_points = np.linspace(start_point, end_point, num_segments - excluded_end)[1:-1]
            combined_array.extend(side_2_points)
            combined_array.extend(array[-1:])
        elif num_segments - excluded_end == 1:
            combined_array.extend(array[-1:])

        filtered_arrays.append(np.array(combined_array))
    print("Total number filtered centroids:", len(filtered_arrays))
    return filtered_arrays
