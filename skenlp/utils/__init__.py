from .seed import set_seeds, get_default_random_seed, set_default_random_seed, set_deterministic_mode,\
    is_deterministic_mode
from .data import combine_datasets, get_dataset, download_file_from_google_drive
from .time import format_time
from .report import classification, classification_report, f1_results, print_message, show_message_with_bar
from .log import get_logger, DumbLogger, SmartLogger
from .device import select_device
from .visualisation import VisualizationDataRecord, text_visualization
from .precision import get_int_precision, get_default_precision


def get_options():
    import argparse
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--train_dataset', type=str, default='moral_together_false')
    parser.add_argument('--test_dataset', type=str, default='moral_together_false')
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--metrics', nargs="+", type=str, default=['acc', 'f1'])
    parser.add_argument('--out_folder', type=str, default='outs')
    parser.add_argument('--log_folder', type=str, default='logs')
    parser.add_argument('--local_exp_file_available', type=str, default='false')
    parser.add_argument('--local_explainer', type=str, default='lrp')
    parser.add_argument('--aggregator', type=str, default='avg_50')
    parser.add_argument('--global_explainer', type=str, default='cart')
    parser.add_argument('--max_skipgrams_length', type=int, default=2)
    parser.add_argument('--run_mode', type=str, default='glob')
    args = parser.parse_args()
    return args


def str2bool(v):
    import argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')