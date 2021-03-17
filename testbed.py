import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


class Test:
    def get_obs(self):
        raise NotImplementedError


class MixedRandomDist(Test):
    def __init__(self, points_per_dist):
        self.points_per_dist = points_per_dist

    def get_obs(self, points_per_dist=None):
        if points_per_dist is not None:
            self.points_per_dist = points_per_dist
        obs, rvs, xlabelss, names, weights = self._generate_random_mixed_dist(
            self.points_per_dist
        )
        others = (rvs, xlabelss, names, weights)
        return obs, others

    def _generate_random_dist_data(self, M=100):
        distr_random_index = np.random.randint(0, 105)
        loc = (np.random.rand(1) - 0.5) * 20  # from -10 to 10
        scale = np.random.rand(1) * 10  # from -10 to 10
        name, params = scipy.stats._distr_params.distcont[distr_random_index]
        params = list(params) + [loc[0], scale[0]]
        dist = getattr(scipy.stats, name)
        frozen_rv = dist(*params)
        sample = frozen_rv.rvs(size=M)
        # Create x label containing the distribution parameters
        p_names = ["loc", "scale"]
        if dist.shapes:
            p_names = [sh.strip() for sh in dist.shapes.split(",")] + ["loc", "scale"]
        xlabels = ", ".join(f"{pn}={pv:.2f}" for pn, pv in zip(p_names, params))
        return sample, frozen_rv, xlabels, name

    def _generate_random_mixed_dist(self, points_per_dist):
        obs, rvs, xlabelss, names = [], [], [], []
        for size in points_per_dist:
            sample, rv, xlabels, name = self._generate_random_dist_data(M=size)
            obs += sorted(sample)
            rvs.append(rv)
            xlabelss.append(xlabels)
            names.append(name)
            weights = [e / np.sum(points_per_dist) for e in points_per_dist]
        return obs, rvs, xlabelss, names, weights


class GrayLevelQuantization:
    def __init__(self, pics_dir="pics", pics_names=None):
        self.pics_dir = pics_dir
        if pics_names is None:
            self.pics_names = sorted(os.listdir(pics_dir))
        else:
            self.pics_names = pics_names

    def load_imgs(self):
        "Be aware of memory consumption!"
        self.imgs_dict = {
            img_name: cv2.imread(os.path.join(self.pics_dir, img_name))
            for img_name in self.pics_names
        }
        return self.imgs_dict

    def get_obs(self):
        self.load_imgs()
        pimgs = {
            img_name: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            for img_name, image in self.imgs_dict.items()
        }
        obs_dict = {img_name: pimgs[img_name].reshape(-1) for img_name in pimgs}
        return obs_dict, pimgs

    def quantize_gray_image(self, grayimg, intervals):
        """Creates a quantization of grayimg by substituting the pixel values in each interval
        by its mean value

        Args:
            img (np.ndarray): dimension 2 array with uint8 type
            intervals (list): list of K non overlapping interval (start, end) tuples
            [(a_0, b_0), ..., (a_K, b_K)]. Must be ordered and non overlapping:
            b_i < a_{i+1} and a_i <= b_i for all i. Intervals are inclusive [start, end].
        """
        assert (
            type(grayimg[0, 0]) == np.uint8
        ), "Input image is not a 2 dimensional uint8 array"
        assert all([True]+[
            intervals[i][1] <= intervals[i + 1][0] for i in range(len(intervals) - 1)
        ]), "Provided intervals are not ordered"
        assert [
            interval[0] <= interval[1] for interval in intervals
        ], "End is before start"

        non_intervals = get_nonintervals(intervals)

        qimg = np.empty_like(grayimg)
        for interval in intervals + non_intervals:
            selected_indices = (interval[0] <= grayimg) & (grayimg <= interval[1])
            if selected_indices.any():
                qimg[selected_indices] = np.mean(grayimg[selected_indices])
            else:
                print(interval)

        return qimg


def get_nonintervals(intervals):
    if intervals[0][0] > 0 and intervals[-1][1] < 255:
        return (
            [(0, intervals[0][0] - 1)]
            + [
                (intervals[i][1] + 1, intervals[i + 1][0] - 1)
                for i in range(len(intervals) - 1)
            ]
            + [(intervals[-1][1] + 1, 255)]
        )

    elif intervals[0][0] > 0:
        return [(0, intervals[0][0] - 1)] + [
            (intervals[i][1] + 1, intervals[i + 1][0] - 1)
            for i in range(len(intervals) - 1)
        ]

    elif intervals[-1][1] < 255:
        return [
            (intervals[i][1] + 1, intervals[i + 1][0] - 1)
            for i in range(len(intervals) - 1)
        ] + [(intervals[-1][1] + 1, 255)]

    else:
        return [
            (intervals[i][1] + 1, intervals[i + 1][0] - 1)
            for i in range(len(intervals) - 1)
        ]


class OrientationMatching:
    def __init__(self, pics_dir="pics", pics_names=None, ksize=1):
        self.ksize = ksize
        self.pics_dir = pics_dir
        if pics_names is None:
            self.pics_names = sorted(os.listdir(pics_dir))
        else:
            self.pics_names = pics_names

    def load_imgs(self):
        "Be aware of memory consumption!"
        self.imgs_dict = {
            img_name: cv2.imread(os.path.join(self.pics_dir, img_name))
            for img_name in self.pics_names
        }
        return self.imgs_dict

    def get_obs(self):
        self.load_imgs()
        pimgs = {
            img_name: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            for img_name, image in self.imgs_dict.items()
        }
        lines_dict = {img_name: self.get_lines(img) for img_name, img in pimgs.items()}
        obs_dict = {
            img_name: self.get_angles(lines) for img_name, lines in lines_dict.items()
        }
        return obs_dict, lines_dict

    def get_lines(self, img):
        # detect lines
        self.fld = cv2.ximgproc.createFastLineDetector(
            _do_merge=True, _length_threshold=min(img.shape[:2]) // 10
        )
        smooth = lambda img: cv2.GaussianBlur(img, (self.ksize, self.ksize), 0)
        return self.fld.detect(smooth(img))

    def angle_fn(self, line):
        if line[2] == line[0]:
            return np.pi / 2
        else:
            return np.arctan((line[3] - line[1]) / (line[2] - line[0]))

    def get_angles(self, lines):
        return [self.angle_fn(line) for line in lines.squeeze()]

    def draw_lines(self, img, lines, angles, intervals):
        intervals = self._solve_discontinuity_problem(intervals)
        result_img = self.fld.drawSegments(img, lines)
        lines_img = self.fld.drawSegments(np.ones_like(img) * 255, lines)
        # create image
        for aline, angle in zip(lines, angles):
            for start, end in intervals:
                if start <= angle <= end:
                    line = np.array(aline).astype(np.int)
                    
                    lines_img = cv2.line(
                        lines_img,
                        tuple(line[0, :2]),
                        tuple(line[0, 2:]),
                        (255, 0, 0),
                        5,
                    )
                    result_img = cv2.line(
                        result_img,
                        tuple(line[0, :2]),
                        tuple(line[0, 2:]),
                        (255, 0, 0),
                        5,
                    )

        # get histogram and bin limits
        angles_hist, bin_edges = np.histogram(angles, bins=180)

        return result_img, lines_img, angles_hist, bin_edges

    def _solve_discontinuity_problem(self, intervals):
        intervalsc = intervals.copy()
        for i in range(len(intervals)):
            a, b = intervals[i]
            if b < a:  # assume it goes aroun the circle
                del intervalsc[i]
                intervalsc = [(-np.pi / 2, b)] + intervalsc + [(a, np.pi / 2)]
        intervals = intervalsc
        return intervals


##########################################################################################
############################################################################################################
######### DEPRECATED
##########################################################################################


def orientation_matching(img, intervals, smoothing_kernel_size):
    ksize = smoothing_kernel_size
    # detect lines
    fld = cv2.ximgproc.createFastLineDetector(
        _do_merge=True, _length_threshold=min(img.shape) // 10
    )
    smooth = lambda img: cv2.GaussianBlur(img, (ksize, ksize), 0)
    lines = fld.detect(smooth(img))
    # get angles
    angle_fn = lambda line: np.arctan((line[3] - line[1]) / (line[2] - line[0])) if (line[2]!=line[0]) else np.pi/2
    angles = [angle_fn(line) for line in lines.squeeze()]

    result_img = fld.drawSegments(img, lines)
    lines_img = fld.drawSegments(np.ones_like(img) * 255, lines)
    # create image
    for aline, angle in zip(lines, angles):
        for start, end in intervals:
            if start <= angle <= end:
                line = np.array(aline).astype(np.int)
                lines_img = cv2.line(
                    lines_img, tuple(line[0, :2]), tuple(line[0, 2:]), (255, 0, 0), 5
                )
                result_img = cv2.line(
                    result_img, tuple(line[0, :2]), tuple(line[0, 2:]), (255, 0, 0), 5
                )

    # get histogram and bin limits
    angles_hist, bin_edges = np.histogram(angles, bins=180)

    #############################################################################
    #############################################################################
    ### if selecting maximum bins
    #############################################################################
    # # get indices of angles in hist
    # angles_hist_indices = np.digitize(angles, bins=bin_edges, right=False) - 1
    # angles_hist_indices[np.argmax(angles_hist_indices)] -= 1
    # # get hist index of maximum
    # max_angle_hist_index = angles_hist.argsort()[-1:][::-1]

    # result_img = fld.drawSegments(pimgs[img_name], lines)
    # lines_img = fld.drawSegments(np.ones_like(pimgs[img_name]) * 255, lines)
    # # create image
    # for i, angle_hist_index in enumerate(angles_hist_indices):
    #     if angle_hist_index == max_angle_hist_index:  # when belonging to gratest bin
    #         lines_img = cv2.line(lines_img,tuple(lines[i][0, :2]), tuple(lines[i][0, 2:]),(255,0,0),1)
    #         result_img = cv2.line(result_img,tuple(lines[i][0, :2]), tuple(lines[i][0, 2:]),(255,0,0),1)

    #############################################################################
    #############################################################################
    ### test behavior of np.digitize and np.histogram
    #############################################################################

    # finished = False
    # while not finished:
    #     n_bins = 5
    #     n_obs = 10
    #     # angles = np.pi / 2 * (np.random.rand(n_obs) * 2 - 1)
    #     angles = sorted(np.random.randint(0, 90, n_obs))
    #     count, bin_edges = np.histogram(angles, bins=n_bins)
    #     indices = np.digitize(angles, bins=bin_edges, right=False)
    #     indices[-1] -= 1
    #     for bin_edge in bin_edges[1:-1]:
    #         if bin_edge in angles:
    #             finished = True

    # print('angles')
    # print(angles)

    # print('-'*20)
    # print('count')
    # print(count)
    # print('-'*20)
    # print('edges')
    # print(bin_edges)

    # print('-'*20)
    # print('indices')
    # print(indices)
    # print('-'*20)
    # print('number of indices')
    # print(len(indices))

    # plt.figure()
    # plt.bar(bin_edges[:-1], count, width=bin_edges[1] - bin_edges[0]-0.5, align='edge')
    # plt.plot(angles, np.zeros(len(angles)), 'x', c=(1, 0, 0))
    # plt.show()

    return result_img, lines_img, angles_hist, bin_edges


# def get_gray_images_hists(pics_dir='pics', pics_names=None):
#     if pics_names is None:
#         pics_names = sorted(os.listdir(pics_dir))
#     imgs = {
#         img_name: cv2.imread(os.path.join(pics_dir, img_name))
#         for img_name in pics_names
#     }
#     pimgs = {img_name: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for img_name, image in imgs.items()}
#     obs_dict = {img_name: pimgs[img_name].reshape(-1) for img_name in pimgs}

#     return obs_dict, pimgs

