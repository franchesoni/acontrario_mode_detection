from testbed import get_nonintervals, orientation_matching
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_distribution(sample, frozen_rv, xlabels, name):
    size = len(sample)
    x = np.linspace(frozen_rv.ppf(0.01), frozen_rv.ppf(0.99), size)

    fig = plt.figure()
    ax = plt.gca()
    plt.hist(sample, int(np.sqrt(len(sample))), density=True)
    ax.plot(x, frozen_rv.pdf(x), c='black', alpha=0.5)
    ax.set_title(name, pad=25)
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel(xlabels, fontsize=8, labelpad=10)
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='both', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

def visualize_mixed_distribution(obs, rvs, names, weights, intervals=None, priorpdf_fn=None, mypdf_fn=None):
    size = len(obs)
    minx, maxx = min(obs)-0.01, max(obs)+0.01  #min(rv.ppf(0.01) for rv in rvs), max(rv.ppf(0.99) for rv in rvs)
    x = np.linspace(minx, maxx, size)
    pdf = np.zeros(size)
    for idx in range(len(rvs)):
        pdf += rvs[idx].pdf(x) * weights[idx]
    name = ' + '.join(f'{weights[idx]:.2f}*{names[idx]}' for idx in range(len(rvs)))
       

    fig = plt.figure()
    ax = plt.gca()
    plt.hist(obs, int(np.sqrt(len(obs))), density=True)
    ax.plot(x,  pdf, c='green', alpha=0.5)
    if priorpdf_fn:
        priorpdf = priorpdf_fn(x)
        ax.plot(x, priorpdf, c='red', alpha=0.2)
    if mypdf_fn:
        mypdf = mypdf_fn(x.reshape(-1, 1))
        ax.plot(x, mypdf, c='blue', alpha=0.2)
    ax.plot(obs, np.zeros(len(obs)), 'x', c='red')
    ax.set_title(name, pad=25)
    ax.set_xlim(x.min(), x.max())
    ax.tick_params(axis='both', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if intervals:
        for start, end in intervals:
            ax.add_patch(patches.Rectangle((start, 0), end-start, pdf.max(), facecolor=(0, 1, 0, 0.3)))


    plt.show()


def visualize_pimgs_and_obs_dict(pimgs, obs_dict):
    for img_name in pimgs:
        plt.figure(figsize=(20, 5))
        plt.title(img_name)
        plt.subplot(1, 2, 1)
        plt.imshow(pimgs[img_name], cmap='gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.hist(obs_dict[img_name], bins=256)
        plt.show()

def show_two_images_and_hists(img, img2, intervals):
    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.hist(img.reshape(-1), range=[0,255], bins=255)
    ax1 = plt.subplot(2, 2, 4)
    ax2 = ax1.twinx()
    ax1.hist(img.reshape(-1), range=[0,255], bins=255, density=True)
    non_intervals = get_nonintervals(intervals)
    for start, end in intervals:
        ax1.add_patch(patches.Rectangle((start, 0), end-start+1, np.histogram(img, range=[0,255], bins=255)[0].max() / img.size, facecolor=(0, 1, 0, 0.3)))
    for start, end in non_intervals:
        ax1.add_patch(patches.Rectangle((start, 0), end-start+1, np.histogram(img, range=[0,255], bins=255)[0].max() / img.size, facecolor=(0.5, 0.5, 0, 0.3)))
    ax2.hist(img2.reshape(-1), range=[0,255], bins=255, density=True, color='r')
    plt.show()


def visualize_hist_and_intervals(ax1, values, intervals, show_non_intervals=True):
    for start, end in intervals:
        color = tuple([1, 0, 0, 0.5 / len(intervals)])
        ax1.add_patch(patches.Rectangle((start, 0), end-start+1, np.histogram(values, range=[0,255], bins=255)[0].max() / values.size, facecolor=color))
    if show_non_intervals:
        non_intervals = get_nonintervals(intervals)
        for start, end in non_intervals:
            color = tuple([1, 0, 0, 0.5 / len(intervals)])
            ax1.add_patch(patches.Rectangle((start, 0), end-start+1, np.histogram(values, range=[0,255], bins=255)[0].max() / values.size, facecolor=(0.5, 0.5, 0, 0.3)))
    ax1.hist(values.reshape(-1), range=[0,255], bins=255, density=True)
 

def visualize_quantization(glq, pimgs, img_name='chicago.jpg', intervals=[(50, 100), (120, 195)]):
    img = pimgs[img_name]
    qimg = glq.quantize_gray_image(img, intervals)
    print('Number of gray tones:', len(set(list(qimg.reshape(-1)))))
    show_two_images_and_hists(img, qimg, intervals)


def visualize_orientation_matching(om, pimgs, img_name='barandas.jpg', intervals=[(0.30,0.38), (0.52, 0.56), (1.534, -1.51)], smoothing_kernel_size=1):
    """ Applies orientaiton matching using provided <intervals>

    Args:
        pimgs (dict): processed images dict
        img_name (str): img key name in pimgs dict
        intervals (list): list of K non overlapping interval (start, end) tuples
          [(a_0, b_0), ..., (a_K, b_K)]. Excepting the one including the extremes,
          must be ordered and non overlapping: b_i < a_{i+1} and a_i <= b_i for all i.
          Intervals are inclusive [start, end]. In radians.
        smoothing_kernel_size (int, optional): Gaussian kernel size. Defaults to 1 (do nothing).
    """
    img = pimgs[img_name]
    intervals = om._solve_discontinuity_problem(intervals)
    result_img, lines_img, angles_hist, bin_edges = orientation_matching(img, intervals, smoothing_kernel_size)

    plt.figure(figsize=(20, 20))
    ax1 = plt.subplot(2, 2, 1)
    plt.bar(bin_edges[:-1], angles_hist, width=bin_edges[1] - bin_edges[0], align='edge')
    for start, end in intervals:
        ax1.add_patch(patches.Rectangle((start, 0), end-start, angles_hist.max(), facecolor=(1, 0, 0, 0.3)))
    plt.xticks([-np.pi / 2, -np.pi/4, 0, np.pi/4, np.pi/2])
    plt.subplot(2, 2, 2)
    plt.imshow(lines_img, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(pimgs[img_name], cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(result_img)
    plt.axis('off')
    plt.show()


def visualize_intervals(meaningful_modes, meaningful_modes_values):
    plt.figure()
    for i in range(len(meaningful_modes)):
        plt.plot(np.arange(meaningful_modes[i][0], meaningful_modes[i][1]+1), np.ones(meaningful_modes[i][1] - meaningful_modes[i][0] + 1) * i, color=(1, 0, 0, meaningful_modes_values[i] / max(meaningful_modes_values)))
    plt.show()

