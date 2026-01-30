import numpy as np
import random

def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def get_FPS(pts, K):
    """
    Farthest point sampling alongside indices
    """
    farthest_pts = np.zeros((K, 3))
    init_random = np.random.randint(len(pts))
    farthest_pts[0] = pts[init_random]
    distances = calc_distances(farthest_pts[0], pts)
    pt_indices = [init_random]
    for i in range(1, K):
        arg_max_ind = np.argmax(distances)
        farthest_pts[i] = pts[arg_max_ind]
        pt_indices.append(arg_max_ind)
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts, pt_indices