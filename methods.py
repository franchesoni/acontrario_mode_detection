import functools
import time
import warnings
warnings.filterwarnings("error")

import numpy as np
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity


def get_pdf_fns(obs, amin, amax):
    # intervals = None; amin, amax = min(obs), max(obs)
    def priorpdf_fn(x, amin, amax):
        prior_value = 1 / (amax - amin)
        result = np.zeros(x.size)
        result[(amin <= x) & (x <= amax)] = prior_value
        return result

    priorpdf_fn = functools.partial(priorpdf_fn, amin=amin, amax=amax)

    kde = KernelDensity(kernel="epanechnikov", bandwidth=1).fit(
        np.array(obs).reshape(-1, 1)
    )
    mypdf_fn = lambda x: np.exp(kde.score_samples(x))
    return priorpdf_fn, mypdf_fn


def H(r_ab, p_ab):
    """Relative entropy as defined in Chapter 4

    Args:
        r_ab (float): estimate of probability from data for the interval [a, b]
        p_ab (float): prior probability for the interval [a, b]

    Returns:
        float: relative entropy
    """
    if r_ab <= p_ab:
        return 0
    elif p_ab == 0:
        return np.inf
    elif r_ab == 0:
        return -np.inf
    else:  # we can make this more robust and efficient
        return r_ab * np.log(r_ab / p_ab) + (1 - r_ab) * np.log((1 - r_ab) / (1 - p_ab))


def H_gap(r_ab, p_ab):
    """Relative entropy as defined in Chapter 4 but as the gaps version.

    Args:
        r_ab (float): estimate of probability from data for the interval [a, b]
        p_ab (float): prior probability for the interval [a, b]

    Returns:
        float: relative entropy
    """
    if p_ab == 0:
        return -np.inf
    elif r_ab == 0:
        return np.inf
    elif r_ab < p_ab:
        return r_ab * np.log(r_ab / p_ab) + (1 - r_ab) * np.log(
            (1 - r_ab) / (1 - p_ab)
        )
    else:  # we can make this more robust and efficient
        return 0


class ContinuousMethod:
    def __init__(self, eps=1, circular=False):
        self.eps = eps
        self.circular = circular

    def get_all_meaningful_intervals(self, sobs, M, d, eps):
        is_meaningful = lambda value: value > np.log(M * (M - 1) / (2 * eps)) / (M - 2)
        mintervals, values = [], []
        for i1 in range(M - 1):
            for i2 in range(i1 + 1, M):
                r_ab = (i2 - i1 - 1) / (
                    M - 2
                )  # number of obs between [sobs[idx1], sobs[idx2]]
                p_ab = (sobs[i2] - sobs[i1]) / d
                value = H(r_ab, p_ab)
                if is_meaningful(value):
                    mintervals.append((i1, i2))
                    values.append(value)
        return mintervals, values

    def get_some_meaningful_intervals_fast(self, sobs, M, d, eps, livess=100):
        kde = KernelDensity(kernel="epanechnikov", bandwidth=1).fit(
            np.array(sobs).reshape(-1, 1)
        )
        pdf_fn = lambda x: np.exp(kde.score_samples(x))
        pdf = pdf_fn(sobs.reshape(-1, 1))
        max_indices = argrelextrema(pdf, np.greater)
        is_meaningful = lambda value: value > np.log(M * (M - 1) / (2 * eps)) / (M - 2)
        mintervals, values = [], []

        # for local maxima
        for max_index in max_indices[0]:
            a, b = max_index-1, max_index+1
            r_ab = (b - a - 1) / (M-2)
            p_ab = (sobs[b] - sobs[a]) / d
            value = H(r_ab, p_ab)
            am, bm = a, b

            finished = False
            lives, c = livess, np.log2(M/2) / livess
            while not finished:
                # expand a and b
                ae, be = max(0, am-int(2**(c*(livess - lives)))), min(bm+int(2**(c*(livess-lives))), len(sobs)-2)
                # try a
                r_abe = (bm - ae - 1) / (M-2)
                p_abe = (sobs[bm] - sobs[ae]) / d
                # check if is better
                valuee = H(r_abe, p_abe)
                    
                if value <= valuee and ((am, bm) != (ae, bm)):
                    am, bm = ae, bm
                    value = valuee
                    continue
        
                # try b                 
                r_abe = (be - am - 1) / (M-2)
                p_abe = (sobs[be] - sobs[am]) / d
                # check if is better
                valuee = H(r_abe, p_abe)
                if value <= valuee and ((am, bm) != (am, be)):
                    am, bm = am, be
                    value = valuee
                    continue

                if is_meaningful(value):
                    mintervals.append([sobs[am], sobs[bm]])
                    values.append(value)

                am, bm = ae, be
                lives -= 1
                finished = lives == 0

        mintervals, values = (
            list(t) for t in zip(*sorted(zip(mintervals, values)))
        )
        return mintervals, values


    def get_some_meaningful_intervals(self, sobs, M, d, eps):
        is_meaningful = lambda value: value > np.log(M * (M - 1) / (2 * eps)) / (M - 2)
        mintervals, values = [], []
        for i1 in range(M - 1):  # for first index
            H_i1i2p1 = 0  # because consecutive points contains 0 points
            ignore_next = False
            for i2 in range(i1 + 1, M - 1):
                H_i1i2 = H_i1i2p1
                r_i1i2p1 = (i2 + 1 - i1 - 1) / (
                    M - 2
                )  # number of obs between [sobs[idx1], sobs[idx2]]
                p_i1i2p1 = (sobs[i2 + 1] - sobs[i1]) / d
                try:
                    H_i1i2p1 = H(r_i1i2p1, p_i1i2p1)
                except RuntimeWarning:
                    if i1 == 0 and i2 == M - 2:
                        H_i1i2p1 = -np.inf
                    else:
                        raise RuntimeError
                if ignore_next:
                    if H_i1i2 <= H_i1i2p1 and is_meaningful(H_i1i2p1):
                        ignore_next = False
                    continue
                if H_i1i2p1 < H_i1i2 and is_meaningful(H_i1i2):
                    mintervals.append([sobs[i1], sobs[i2]])
                    values.append(H_i1i2)
                    ignore_next = True
            H_i1i2 = H_i1i2p1
            if ignore_next or (i1 == 0 and i2 == M - 2):  # or support
                continue
            elif is_meaningful(H_i1i2):
                mintervals.append([sobs[i1], sobs[i2 + 1]])  # last one
                values.append(H_i1i2)
        return mintervals, values

    def get_some_meaningful_gaps(self, sobs, M, d, eps=1):
        """This function returns meaningful gaps but not all of them, as it
        discards some of the ones that include any other.
        """
        is_meaningful = lambda value: value > np.log(M * (M + 1) / (2 * eps)) / (M - 2)
        mgaps, values = [], []
        for i1 in range(M - 1):  # for first index
            H_i1i2p1 = 0  # consecutive points can't be meaningful gaps
            ignore_next = False
            for i2 in range(i1 + 1, M - 1):
                H_i1i2 = H_i1i2p1
                r_i1i2p1 = (i2 + 1 - i1 - 1) / (
                    M - 2
                )  # number of obs between [sobs[idx1], sobs[idx2]]
                p_i1i2p1 = (sobs[i2 + 1] - sobs[i1]) / d
                H_i1i2p1 = H_gap(r_i1i2p1, p_i1i2p1)
                if ignore_next:
                    if H_i1i2 <= H_i1i2p1 and is_meaningful(H_i1i2p1):
                        ignore_next = False
                    continue
                if H_i1i2p1 < H_i1i2 and is_meaningful(H_i1i2):
                    mgaps.append([sobs[i1], sobs[i2]])
                    values.append(H_i1i2)
                    ignore_next = True
            H_i1i2 = H_i1i2p1
            if ignore_next:
                continue
            elif is_meaningful(H_i1i2):
                mgaps.append([sobs[i1], sobs[i2 + 1]])  # last one
                values.append(H_i1i2)
        return mgaps, values

    def get_meaningful_modes(self, meaningful_intervals, values, meaningful_gaps):
        meaningful_modes = meaningful_intervals.copy()
        mode_values = values.copy()
        indices_to_remove = []
        gaps_init = 0
        for idx, (ainterval, binterval) in enumerate(meaningful_modes):
            for idxgap, (agap, bgap) in enumerate(meaningful_gaps[gaps_init:]):
                if ainterval <= agap <= bgap <= binterval:  # if contains gap
                    indices_to_remove.append(idx)
                    gaps_init = idxgap
                    break
                elif binterval < agap:
                    break
        indices_to_remove = sorted(indices_to_remove)[::-1]  # larger first
        for idx_to_remove in indices_to_remove:
            del meaningful_modes[idx_to_remove]
            del mode_values[idx_to_remove]
        return meaningful_modes, mode_values

    def take_maximal_meaningful_modes(self, meaningful_intervals, values):
        intervals = meaningful_intervals
        assert intervals == sorted(intervals)
        svalues, sintervals = (
            list(t)[::-1] for t in zip(*sorted(zip(values, intervals)))
        )  # sorted by value

        for idx1 in reversed(range(len(svalues))):
            included = False
            idx2 = 0
            while idx2 < idx1 and not included:
                included = self.intersects(sintervals[idx1], sintervals[idx2])
                idx2 += 1  # next comparison
            if included:
                del sintervals[idx1]
                del svalues[idx1]
        intervals, values = (
            list(t) for t in zip(*sorted(zip(sintervals, svalues)))
        )  
        return intervals, values

    def intersects(self, interval1, interval2):
        try:
            assert (interval1 == sorted(interval1)) and (interval2 == sorted(interval2))
        except AssertionError:
            print(interval1)
            print(sorted(interval1))
            print(interval2)
            print(sorted(interval2))
            breakpoint()
            raise AssertionError("mmm")

        a1, b1 = interval1
        a2, b2 = interval2
        return b1 >= a2 and b2 >= a1

    def __call__(self, obs, fast=True):
        """Maximum meaningful mode detector from continuous data.

        Args:
            obs (list): list of real observations
            eps (int, optional): Level of meaningfulness (max NFA). Defaults to 1.
        """
        # assert not is_discrete(obs), "I think you should be using DiscreteMethod"
        sobs = np.array(sorted(obs))
        M = len(sobs)

        start = time.time()
        min_estimate, max_estimate = self.max_min_estimates(sobs, M)
        if self.circular:
            min_estimate, max_estimate = -np.pi, np.pi
            sobs = np.array(
                sorted(np.concatenate((sobs, sobs + np.pi)))
            )  # double data (easy, inefficient way)
            M *= 2
        d = max_estimate - min_estimate
        if fast:
            mintervals, mintervals_values = self.get_some_meaningful_intervals_fast(
                sobs, M, d, self.eps
            )
        else:
            mintervals, mintervals_values = self.get_some_meaningful_intervals(
                sobs, M, d, self.eps
            )

        mgaps, gaps_values = self.get_some_meaningful_gaps(sobs, M, d, eps=1)
        meaningful_modes, meaningful_modes_values = self.get_meaningful_modes(
            mintervals, mintervals_values, mgaps
        )
        meaningful_modes, meaningful_modes_values = (
            list(t) for t in zip(*sorted(zip(meaningful_modes, meaningful_modes_values)))
        )  
        max_meaningful_modes, values = self.take_maximal_meaningful_modes(
            meaningful_modes, meaningful_modes_values
        )
        if self.circular:
            max_meaningful_modes = self.clean_circle(max_meaningful_modes)
        end = time.time()
        print(end - start)
        return max_meaningful_modes, min_estimate, max_estimate

    def clean_circle(self, max_meaningful_modes):
        max_meaningful_modes = (
            np.array(max_meaningful_modes) + np.pi / 2
        ) % np.pi - np.pi / 2
        extreme = [[a, b] for a, b in max_meaningful_modes if b < a]
        if extreme:
            ind_to_delete = []
            for ind, mmm in enumerate(max_meaningful_modes):
                if mmm[0] <= extreme[0][1]:
                    ind_to_delete.append(ind)
                elif extreme[0][0] < mmm[1]:
                    ind_to_delete.append(ind)
            for ind in sorted(ind_to_delete)[::-1]:
                del max_meaningful_modes[ind]
        max_meaningful_modes = sorted(set(map(tuple, max_meaningful_modes)))
        return max_meaningful_modes

    def max_min_estimates(self, sobs, N):
        amin = (N * sobs[0] - sobs[-1]) / (N - 1)
        amax = (N * sobs[-1] - sobs[0]) / (N - 1)
        return amin, amax


def is_discrete(obs):
    """We say an array of observations is discrete if is at least a 10% of
    points are repeated (as it's unlikely continuous values are repeated).

    Args:
        obs (list or np.array): list of real or integer observations

    Returns:
        bool: flag
    """
    return len(set(obs)) < len(obs) - len(obs) // 10


class DiscreteMethod:
    def __init__(self, eps=1):
        self.eps = eps

    def get_gbins(self, obs):
        """Gets gbins of discrete observations. This is basically the set of values
        taken by the observations.

        Args:
            obs (list): discrete list of observations
        """
        assert is_discrete(obs), "Data is not discrete"
        Nbins = len(set(obs))  # number of bins or L
        bins_indices = np.arange(Nbins)  # indices [1, ..., L]
        bins_values = sorted(set(obs))
        obs_indices, bins_count = np.empty_like(obs), np.empty(Nbins)
        for index, value in zip(bins_indices, bins_values):
            coincidences = obs == value
            obs_indices[coincidences] = bins_indices[index]
            bins_count[index] = np.sum(coincidences)
        return obs_indices, bins_indices, bins_values, bins_count

    def get_all_meaningful_intervals(self, bins_indices, bins_count, M, L, eps):
        meaningful_intervals, values = [], []
        is_meaningful = lambda value: value > np.log(L * (L + 1) / (2 * eps)) / M
        for a in bins_indices:
            for b in bins_indices[a:]:
                r_ab = np.sum(bins_count[a : b + 1]) / M
                p_ab = (b - a + 1) / L  # uniform distribution
                value = H(r_ab, p_ab)
                if is_meaningful(value):
                    meaningful_intervals.append([a, b])
                    values.append(value)
        return meaningful_intervals, values

    def get_meaningful_modes(self, meaningful_intervals, values, meaningful_gaps):
        meaningful_modes = meaningful_intervals.copy()
        mode_values = values.copy()
        indices_to_remove = []
        for idx, (ainterval, binterval) in enumerate(meaningful_modes):
            for agap, bgap in meaningful_gaps:
                if ainterval <= agap <= bgap <= binterval:  # if contains gap
                    indices_to_remove.append(idx)
                    break
        indices_to_remove = sorted(indices_to_remove)[::-1]  # larger first
        for idx_to_remove in indices_to_remove:
            del meaningful_modes[idx_to_remove]
            del mode_values[idx_to_remove]
        return meaningful_modes, mode_values

    def get_meaningful_gaps(self, bins_indices, bins_count, M, L, eps):
        """This function returns meaningful gaps but not all of them, as it
        discards some of the ones that include any other.
        """
        meaningful_gaps, values = [], []
        is_meaningful = lambda value: value > np.log(L * (L + 1) / (2 * eps)) / M
        for a in bins_indices:
            H_abp1 = H_gap(np.sum(bins_count[a : a + 1]) / M, 1 / L)
            ignore_next = False
            for b in bins_indices[a:-1]:
                H_ab = H_abp1
                H_abp1 = H_gap(np.sum(bins_count[a : b + 2]) / M, (b + 2 - a) / L)
                if ignore_next:
                    if H_ab <= H_abp1 and is_meaningful(H_abp1):
                        ignore_next = False
                    continue
                if H_abp1 <= H_ab and is_meaningful(H_ab):
                    meaningful_gaps.append([a, b])
                    values.append(H_ab)
                    ignore_next = True
            H_ab = H_abp1
            if ignore_next:
                continue
            elif is_meaningful(H_ab):
                meaningful_gaps.append([a, b + 1])
                values.append(H_ab)
        return meaningful_gaps, values

    def take_maximal_meaningful_modes(self, meaningful_intervals, values):
        intervals = meaningful_intervals
        assert intervals == sorted(intervals)
        svalues, sintervals = (
            list(t)[::-1] for t in zip(*sorted(zip(values, intervals)))
        )  # sorted by value
        for idx1 in reversed(range(len(svalues))):
            included = False
            idx2 = 0
            while idx2 < idx1 and not included:
                included = self.intersects(sintervals[idx1], sintervals[idx2])
                idx2 += 1  # next comparison
            if included:
                del sintervals[idx1]
                del svalues[idx1]
        intervals, values = (
            list(t) for t in zip(*sorted(zip(sintervals, svalues)))
        )  
        return intervals, values

    def intersects(self, interval1, interval2):
        assert (interval1 == sorted(interval1)) and (interval2 == sorted(interval2))
        a1, b1 = interval1
        a2, b2 = interval2
        return b1 >= a2 and b2 >= a1

    def get_impure_meaningful_intervals(self, bins_indices, bins_count, M, L, eps):
        """This function returns meaningful intervals but not all of them, as it
        discards some of the ones that are not maximal meaningful modes.

        Args:
            bins_indices ([type]): [description]
            bins_count ([type]): [description]
            M ([type]): [description]
            L ([type]): [description]
            eps ([type]): [description]

        Returns:
            [type]: [description]
        """
        meaningful_intervals, values = [], []
        is_meaningful = lambda value: value > np.log(L * (L + 1) / (2 * eps)) / M
        for a in bins_indices:
            H_abp1 = H(np.sum(bins_count[a : a + 1]) / M, 1 / L)
            ignore_next = False
            for b in bins_indices[a:-1]:
                H_ab = H_abp1
                H_abp1 = H(np.sum(bins_count[a : b + 2]) / M, (b + 2 - a) / L)
                if ignore_next:
                    if H_ab <= H_abp1 and is_meaningful(H_abp1):
                        ignore_next = False
                    continue
                if H_abp1 < H_ab and is_meaningful(H_ab):
                    meaningful_intervals.append([a, b])
                    values.append(H_ab)
                    ignore_next = True
            H_ab = H_abp1
            if ignore_next:
                continue
            elif is_meaningful(H_ab):
                meaningful_intervals.append([a, b + 1])
                values.append(H_ab)
        return meaningful_intervals, values

    def __call__(self, obs):
        obs_indices, bins_indices, bins_values, bins_count = self.get_gbins(
            obs
        )  # bin data if discrete
        M, L = len(obs), len(bins_indices)
        (
            meaningful_intervals,
            meaningful_intervals_values,
        ) = self.get_impure_meaningful_intervals(
            bins_indices, bins_count, M, L, self.eps
        )
        meaningful_gaps, gaps_values = self.get_meaningful_gaps(
            bins_indices, bins_count, M, L, eps=1
        )
        meaningful_modes, meaningful_modes_values = self.get_meaningful_modes(
            meaningful_intervals, meaningful_intervals_values, meaningful_gaps
        )

        max_meaningful_modes, values = self.take_maximal_meaningful_modes(
            meaningful_modes, meaningful_modes_values
        )
        return [[bins_values[a], bins_values[b]] for a, b in max_meaningful_modes]


def method1(obs, eps=1):
    """Maximum meaningful interval detector for discrete data. Approx 1 second
    with 200x200 image.

    Args:
        obs (list): list of discrete observations
    """
    assert is_discrete(obs), "Data is not discrete"
    obs_indices, bins_indices, bins_values, bins_count = get_gbins(
        obs
    )  # bin data if discrete
    M, L = len(obs), len(bins_indices)
    meaningful_intervals, meaningful_intervals_values = get_impure_meaningful_intervals(
        bins_indices, bins_count, M, L, eps
    )
    meaningful_gaps, gaps_values = get_meaningful_gaps(
        bins_indices, bins_count, M, L, eps
    )
    meaningful_modes, meaningful_modes_values = get_meaningful_modes(
        meaningful_intervals, meaningful_intervals_values, meaningful_gaps
    )

    max_meaningful_modes, values = take_maximal_meaningful_modes(
        meaningful_modes, meaningful_modes_values
    )
    return max_meaningful_modes


# def direct_clean(intervals, values):
#     """Removes those intervals such that the next one is bigger to the right
#     and it's value is as big

#     Args:
#         intervals ([type]): [description]
#         values ([type]): [description]
#     """
#     assert intervals == sorted(intervals)  # ensure all is ok
#     indices_to_remove  = []
#     for idx in range(len(intervals) - 1):
#         if (intervals[idx][0] == intervals[idx+1][0]):
#             if values[idx] <= values[idx+1]:
#                 indices_to_remove.append(idx)
#             elif values[idx] > values[idx+1]:
#                 indices_to_remove.append(idx)
#     for idx in sorted(list(set(indices_to_remove)))[::-1]:
#         del intervals[idx]
#         del values[idx]

#     zip(*sorted(zip(list1, list2)))

#     assert intervals == sorted(intervals)
#     indices_to_remove  = []
#     for idx in range(len(intervals) - 1):
#         if (intervals[idx][0] == intervals[idx+1][0]):
#             if values[idx] <= values[idx+1]:
#                 indices_to_remove.append(idx)
#             elif values[idx] > values[idx+1]:
#                 indices_to_remove.append(idx)
#     for idx in sorted(list(set(indices_to_remove)))[::-1]:
#         del intervals[idx]
#         del values[idx]


#     assert intervals == sorted(intervals)
#     indices_to_remove = []
#     for idx in range(len(intervals) - 1)
#     return intervals, values
# def take_maximal_meaningful_modes(meaningful_intervals, values):
#     meaningful_intervals, values = direct_clean(meaningful_intervals, values)

#     finished = False
#     print('-'*20)
#     while not finished:
#         indexed_pairs = list(itertools.combinations(enumerate(meaningful_intervals), 2))
#         for pair in indexed_pairs:
#             idx, (a, b) = pair[0]
#             idx2, (a2, b2) = pair[1]
#             if b < a2 or b2 < a:  # non overlapping
#                 finished = True  # if exits the loop without breaking it's finished
#                 continue
#             else:  # one contained in the other or intersecting: delete lower value
#                 idxx = get_lower_value_index(values, idx, idx2, b, a, b2, a2)
#                 del meaningful_intervals[idxx]
#                 del values[idxx]
#                 finished = False  # if it breaks somewhere it does not finish
#                 break
#     return meaningful_intervals, values

# def get_lower_value_index(values, idx, idx2, b, a, b2, a2):
#     if values[idx] < values[idx2]:
#         idxx = idx
#     elif values[idx] != values[idx2]:
#         idxx = idx2
#     else:
#         idxx = idx if (b - a > b2 - a2) else idx2
#     return idxx

# idx1 = 0
# while idx1 < len(svalues)-1:
#     indices_to_remove = []
#     for idx2 in range(idx1+1, len(svalues)):
#         if intersects(sintervals[idx2], sintervals[idx2]):
#             if (svalues[idx2] < svalues[idx1]):
#                 indices_to_remove.append(idx2)
#             elif (svalues[idx1] == svalues[idx2]) and (sintervals[idx1][1] - sintervals[idx1][0] > sintervals[idx2][1] - sintervals[idx2][0]):
#                 indices_to_remove.append(idx2)
#             elif (svalues[idx1] == svalues[idx2]) and (sintervals[idx1][1] - sintervals[idx1][0] < sintervals[idx2][1] - sintervals[idx2][0]):
#                 indices_to_remove.append(idx1)
#                 idx1 -= 1
#                 break
#             else:
#                 RuntimeError("This shouln't happen")
#     for idx in sorted(indices_to_remove)[::-1]:
#         del svalues[idx]
#         del sintervals[idx]
#     idx1 += 1

# from sklearn.model_selection import GridSearchCV
# # use grid search cross-validation to optimize the bandwidth
# params = {'bandwidth': np.logspace(-1, 3, 50)}
# grid = GridSearchCV(KernelDensity(kernel='epanechnikov'), params)
# grid.fit(np.array(obs).reshape(-1, 1))
# mypdf_fn = lambda x: np.exp(stimator_.score_samples(x))

