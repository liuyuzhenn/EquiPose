import numpy as np
import os


def error_auc(errors, thresholds=[5, 10, 20], return_array=False):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []

    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    if return_array:
        return np.array([auc*100 for auc in aucs])
    else:
        return {f'auc@{t}': float(auc)*100 for t, auc in zip(thresholds, aucs)}


def error_acc(errors, thresholds=[5, 10, 20], return_array=False):
    total = len(errors)
    if return_array:
        return np.array([np.sum(errors <= t)/total*100 for t in thresholds])
    else:
        return {f'acc@{t}': np.sum(errors<=t)/total*100 for t in thresholds}


def get_results(folder):
    match_methods = ['sift','superpoint','superglue','loftr']
    subfolders = [f for f in os.listdir(folder) if os.path.isdir(
        os.path.join(folder, f)) and f.lower() in match_methods]
    subfolders.sort()
    subfolders2 = [f for f in os.listdir(folder) if os.path.isdir(
        os.path.join(folder, f)) and f.lower() not in match_methods]
    subfolders2.sort()
    subfolders += subfolders2

    errors_all_R = {}
    errors_all_t = {}

    for sf in subfolders:
        path = os.path.join(folder, sf, 'data.npz')
        if not os.path.exists(path):
            continue
        data = np.load(path)
        errors_R = data['errors_R']
        errors_all_R[sf] = errors_R
        errors_t = data['errors_t']
        errors_all_t[sf] = errors_t
    return errors_all_R, errors_all_t


def result_pretty_format(metrics: dict, head='=', sep='|', pad=10):
    lines = []

    method_lens = np.array([len(k) for k in metrics.keys()])
    pad_method = np.max(method_lens)+1
    for method, metric in metrics.items():
        l = method.center(pad_method)
        for metric_name, data in metric.items():
            l += sep
            if isinstance(data, float):
                l += '{:.2f}'.format(data).center(pad)
            elif isinstance(data, str) or isinstance(data, int):
                l += '{}'.format(data).center(pad)
        lines.append(l)

    ###########
    # heading #
    ###########
    l = 'Method'.center(pad_method)
    for k in metric.keys():
        l += sep
        l += k.center(pad)

    ruler = len(l)*head
    lines = [ruler, l, ruler] + lines + [ruler]

    return lines


