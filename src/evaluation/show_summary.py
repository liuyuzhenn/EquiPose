import argparse
import numpy as np
import os
import sys
import argparse
sys.path.append('.')
from src.evaluation.utils import get_results, result_pretty_format, error_auc, error_acc

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str,
                    default="./evaluation/scannet", help='Data folder')
parser.add_argument('--metric', type=str, default="auc", help='acc,auc')
parser.add_argument('--pad', type=int, default=10, help='Grid size')
parser.add_argument('--head', type=str, default='=', help='Head symbol')
parser.add_argument('--sep', type=str, default='|', help='Separator')
parser.add_argument('-tr', '--thresholds_R', type=str,
                    default='3,5,10', help='Rotation Thresholds')
parser.add_argument('-tt', '--thresholds_t', type=str,
                    default='0.1,0.2,0.5', help='Translation Thresholds')
args = parser.parse_args()


thresholds_R = [float(t) for t in args.thresholds_R.split(',')]
thresholds_t = [float(t) for t in args.thresholds_t.split(',')]
data_folder = args.folder
pad = args.pad
head = args.head
sep = args.sep
m = args.metric

metric_funcs = {
    'acc': error_acc,
    'auc': error_auc,
}


errors_all_R, errors_all_t = get_results(data_folder)

metrics_all = {}
for method in errors_all_R.keys():
    errors_R = errors_all_R[method]
    errors_t = errors_all_t[method]
    if len(errors_R)==0: continue

    total = len(errors_R)
    mask = ~np.isinf(errors_R)
    errors_R = errors_R[mask]

    total = len(errors_t)
    mask = ~np.isinf(errors_t)
    errors_t = errors_t[mask]

    metrics = {}
    
    m1 = metric_funcs[m](errors_R, thresholds_R)
    m1 = {'R_'+k:v for k,v in m1.items()}
    m2 = metric_funcs[m](errors_t, thresholds_t)
    m2 = {'t_'+k:v for k,v in m2.items()}

    metrics.update(m1)
    metrics.update(m2)
    m3 = {
        'mn_R': float(np.mean(errors_R)),
        'mn_t': float(np.mean(errors_t)),
    }
    metrics.update(m3)
    metrics_all[method] = metrics


lines = result_pretty_format(metrics_all, head=head, sep=sep, pad=pad)
for l in lines:
    print(l)
