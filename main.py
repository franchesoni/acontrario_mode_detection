import matplotlib.pyplot as plt
import numpy as np

from methods import ContinuousMethod, DiscreteMethod, get_pdf_fns, method1
from testbed import GrayLevelQuantization, MixedRandomDist, OrientationMatching
from visualization import (
    visualize_mixed_distribution,
    visualize_orientation_matching,
    visualize_quantization,
)

SEED = 42
np.random.seed(SEED)
if __name__ == "__main__":
    ###########################################################################
    ### Work with synthetic histograms
    ###########################################################################
    for _ in range(4):
        continuous = ContinuousMethod(eps=1)
        points_per_dist = [400] * 4
        mdst = MixedRandomDist(points_per_dist)
        obs, others = mdst.get_obs()
        rvs, xlabelss, names, weights = others
        intervals, amin, amax = continuous(obs)
        # intervals, amin, amax = None, min(obs), max(obs)
        priorpdf_fn, mypdf_fn = get_pdf_fns(obs, amin, amax)
        ###########################################################################
        visualize_mixed_distribution(obs, rvs, names, weights, intervals, priorpdf_fn, mypdf_fn)
        ###########################################################################

    ###########################################################################
    ### Work with images (discrete)
    ###########################################################################
    discrete = DiscreteMethod(eps=1)
    glq = GrayLevelQuantization()
    obs_dict, pimgs = glq.get_obs()
    for img_name in list(pimgs.keys()):
        intervals = discrete(obs_dict[img_name])
        ###########################################################################
        visualize_quantization(glq, pimgs, img_name=img_name, intervals=intervals)
        ###########################################################################

    ###########################################################################
    ### Work with images (continuous)
    ###########################################################################
    glq = GrayLevelQuantization()
    obs_dict, pimgs = glq.get_obs()  # use this to load images
    continuous = ContinuousMethod(eps=1, circular=True)
    for img_name in list(pimgs.keys()):
        om = OrientationMatching()
        obs_dict, lines_dict = om.get_obs()
        intervals, amin, amax = continuous(obs_dict[img_name], fast=1000<len(obs_dict[img_name]))
        ###########################################################################
        result_img, lines_img, angles_hist, bin_edges = om.draw_lines(
            pimgs[img_name], lines_dict[img_name], obs_dict[img_name], intervals=intervals
        )
        visualize_orientation_matching(om, pimgs, img_name, intervals=intervals)
        ###########################################################################
