import os
import argparse
from src.evaluation.pipeline import EvaluationLearningBased
from src.config import DictAction, update_configs
from src import load_configs


parser = argparse.ArgumentParser()
parser.add_argument('-w', '--workspace', type=str,
                    default='', help='workspace folder')
parser.add_argument('--dataset', type=str, required=True, help='path of dataset config file')
parser.add_argument(
    '-o',
    '--cfg-options',
    nargs='+',
    action=DictAction,
    help='override some settings in the used config, the key-value pair '
    'in xxx=yyy format will be merged into config file. If the value to '
    'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
    'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
    'Note that the quotation marks are necessary and that no white space '
    'is allowed.')
args = parser.parse_args()

workspace = args.workspace

configs = {
    "workspace": workspace,
    "split": 'test',
}

if args.cfg_options is not None:
    update_configs(configs, args.cfg_options)

dataset = args.dataset
if dataset != "":
    dataset_configs = load_configs(args.dataset)['dataset_configs']
else:
    dataset_configs = load_configs(args.cfg_options['cfg_path'])['dataset_configs']

dataset_configs = load_configs(args.dataset)['dataset_configs']
dataset_configs['batch_size'] = 1
dataset_configs['num_workers'] = 4

print('workspace: {}'.format(workspace))

evaluator = EvaluationLearningBased(dataset_configs, configs)
evaluator.evaluate()
