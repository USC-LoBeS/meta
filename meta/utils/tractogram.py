import numpy as np


def reorient_streamlines(m_centroid, s_centroids):
    """
    Reorients the subject centroids based on the model centroid.

    Parameters:
        m_centroid: Model centroid
        s_centroids: List of subject centroids
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

        return (start_distance < end_distance) and (start < end)


    # Ensure inputs are lists of arrays:
    if isinstance(s_centroids, np.ndarray) and s_centroids.ndim == 2:
        s_centroids = [s_centroids]
    if isinstance(m_centroid, np.ndarray) and m_centroid.ndim == 2:
        m_centroid = [m_centroid]
        
    oriented_s_centroids = []
    for s_centroid in s_centroids:
        if is_flipped(m_centroid, s_centroid):
            oriented_s_centroids.append(s_centroid[::-1])
        else:
            oriented_s_centroids.append(s_centroid)

    return oriented_s_centroids