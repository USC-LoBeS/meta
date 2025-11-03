import sys
import logging
import numpy as np
import nibabel as nib
from collections import defaultdict
from tslearn.metrics import dtw_path
from meta.io.streamline import read_streamlines
from meta.io.transform import load_transformation
from meta.utils.tractogram import reorient_streamlines
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import length, transform_streamlines


logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8',
    level=logging.INFO,
    force=True
)


def get_centroids(streamlines, n_points=500, threshold=np.inf, n_centroids=None):
    """
    Get centroids from a set of streamlines.

    Parameters
    ----------
    streamlines : A list of streamlines.
    n_points : Number of points to use for resampling the centroids.
    threshold : Distance threshold for clustering.
    n_centroids : Number of centroids to return.

    Returns
    -------
    centroids : A list of centroids.
    """

    feature = ResampleFeature(nb_points=n_points)
    metric = AveragePointwiseEuclideanMetric(feature)
    qb = QuickBundles(threshold=threshold, metric=metric, max_nb_clusters=n_centroids)
    centroids = qb.cluster(streamlines).centroids
    logging.info("Average length of centroids: {}".format(np.mean([length(c) for c in centroids])))
    
    return centroids


def get_alignment(model, subject, n_segments, mask_img, transform=None, inverse=False):
    """
    Get the alignment between model and subject streamlines.

    Parameters
    ----------
    model: Path to the model bundle.
    subject: Path to the subject bundle.
    n_segments: Number of segments for alignment.
    mask_img: Path to the mask image.
    transform: Path to the transformation matrix.
    inverse: Whether to invert the transformation matrix.

    Returns
    -------
    updated_corres: List of updated correspondences using DTW.
    original_indices: Indices of DTW points.
    n_segments: Number of segments after alignment.

    """

    ## Reference image:
    ref_image = nib.load(mask_img)

    ## Transform model to subject space:
    model_streamlines, _, _, _ = read_streamlines(model)
    if transform is not None:
        transformation_matrix = load_transformation(transform)
        if inverse:
            transformation_matrix = np.linalg.inv(transformation_matrix)
            print("Inverted transformation matrix for model streamlines.")
        transformed_model = transform_streamlines(model_streamlines, transformation_matrix)
        transformed_model = transform_streamlines(transformed_model, np.linalg.inv(ref_image.affine))
    else:
        transformed_model = transform_streamlines(model_streamlines, np.linalg.inv(ref_image.affine))
    m_centroid = get_centroids(transformed_model, n_points=n_segments, threshold=np.inf)

    ## Subject bundle:
    subject_streamlines, _, _, _ = read_streamlines(subject)
    transformed_subject = transform_streamlines(subject_streamlines, np.linalg.inv(ref_image.affine))
    s_centroid = get_centroids(transformed_subject, n_points=500, threshold=np.inf)

    num_clusters = 100
    centroids = get_centroids(transformed_subject, n_points=500, threshold=2., n_centroids=num_clusters)

    ## Check if the subject centroids are flipped compared to the model:
    s_centroid = reorient_streamlines(s_centroid, m_centroid)
    centroids = reorient_streamlines(centroids, m_centroid)

    ## Compute the correspondence between model and subject centroids:
    dtw_corres = []
    for m, s in zip(m_centroid, s_centroid):
        pathDTW, similarityScore = dtw_path(m, s)
        logging.info(f"DTW similarity score: {similarityScore}")

        model_to_subject = defaultdict(list)
        subject_to_model = defaultdict(list)
        for mi, si in pathDTW:
            model_to_subject[mi].append(si)
            subject_to_model[si].append(mi)

        kept_pairs = set()
        for subject_idx, model_idxs in subject_to_model.items():
            if len(model_idxs) > 1:
                kept_pairs.discard((model_idxs[0], subject_idx))

        for model_idx, subject_idxs in model_to_subject.items():
            if len(subject_idxs) > 1:
                kept_pairs.add((model_idx, subject_idxs[len(subject_idxs) // 2]))

        for model_idx, subject_idxs in model_to_subject.items():
            if len(subject_idxs) == 1:
                subject_idx = subject_idxs[0]
                if len(subject_to_model[subject_idx]) == 1:
                    kept_pairs.discard((model_idx, subject_idx))

        centroid_corres = np.full((n_segments, 3), np.nan, dtype=float)
        for model_idx, subject_idx in kept_pairs:
            centroid_corres[model_idx] = s[subject_idx]
        dtw_corres.append(centroid_corres)
    
    original_indices = np.where(~np.isnan(dtw_corres[0]).all(axis=1))[0]
    n_segments = len(original_indices)

    logging.info(f"Dynamic time warping (DTW) Correspondence shape: {dtw_corres[0].shape}")
    logging.info(f"Original indices of kept points: {original_indices}")
    logging.info(f"Number of updated segments: {n_segments}")

    subject_corres = []
    subject_ref = np.squeeze(dtw_corres)
    subject_ref = subject_ref[~np.isnan(subject_ref).all(axis=1)]
    for idx, centroid in enumerate(centroids):

        centroid = np.squeeze(centroid)
        pathDTW, similarityScore = dtw_path(subject_ref, centroid)
        x1, y1, z1 = subject_ref[:, 0], subject_ref[:, 1], subject_ref[:, 2]
        x2, y2, z2 = centroid[:, 0], centroid[:, 1], centroid[:, 2]
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
        subject_corres.append(np.array(centroid_corres))

    combined_corres = np.stack([subject_ref] + subject_corres, axis=0)
    logging.info(f"Combined Correspondence shape: {combined_corres.shape}")

    ## Remove centroids that are shorter than the average length
    data = []
    for streamline in combined_corres:
        data.append(length(streamline))  
    mean_length = np.mean(data)
    logging.info(f"Average streamlines length: {mean_length}")
    indices = np.where(data < mean_length)
    final_corres = [sl for idx, sl in enumerate(combined_corres) if idx not in indices[0]]
    logging.info(f"Final Correspondence shape: {np.array(final_corres).shape}")


    ## Compute pairwise distances:
    corresponding_points = np.array(final_corres)
    pairwise_distances = np.zeros((corresponding_points.shape[1], corresponding_points.shape[0], corresponding_points.shape[0]))
    for i in range(corresponding_points.shape[1]):
        for j in range(corresponding_points.shape[0]):
            for k in range(j + 1, corresponding_points.shape[0]):
                pairwise_distances[i, j, k] = np.linalg.norm(corresponding_points[j, i] - corresponding_points[k, i])
    pairwise_distances[pairwise_distances == 0] = np.nan
    std_distances = np.nanstd(pairwise_distances, axis=(1, 2))
    excluded_idx = np.where(std_distances <= 3.5)[0]
    logging.info(f"Excluded indices based on std distances: {excluded_idx}")

    if excluded_idx.size > 0:
        
        excluded_start = excluded_idx[0]
        excluded_end = excluded_idx[-1]
        updated_corres = []
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
            if n_segments - excluded_end > 1:
                start_point = array[excluded_end]
                end_point = array[-1]
                side_2_points = np.linspace(start_point, end_point, n_segments - excluded_end)[1:-1]
                combined_array.extend(side_2_points)
                combined_array.extend(array[-1:])
            elif n_segments - excluded_end == 1:
                combined_array.extend(array[-1:])

            updated_corres.append(np.array(combined_array))
    else:
        updated_corres = final_corres
    print("Shape of updated correspondences:", np.array(updated_corres).shape)

    return updated_corres, original_indices, n_segments

