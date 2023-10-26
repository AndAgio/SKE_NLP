import time
import warnings

from skenlp.models import BertConfig, BertForSequenceClassification
from skenlp.data import get_label_names
from skenlp.bin import Explainer
from skenlp.utils import set_seeds, get_logger, get_options, format_time, str2bool


def explain(arguments):
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    set_seeds(seed_val)
    logger = get_logger(arguments=arguments, mode='smart')
    # logger_func = print

    logger.print_it('Explaining {} over {} using {} as local explainer, {} as aggregator and {} as global explainer. '
                    'Test is performed over {}'.format(arguments.model.upper(),
                                                       arguments.train_dataset.upper(),
                                                       arguments.local_explainer.upper(),
                                                       arguments.aggregator.upper(),
                                                       arguments.global_explainer.upper(),
                                                       arguments.test_dataset.upper()))
    model = BertForSequenceClassification(BertConfig(),
                                          num_labels=len(get_label_names(arguments.train_dataset)),
                                          checkpoint='{}/models/best_{}_{}.pt'.format(arguments.out_folder,
                                                                                      arguments.model,
                                                                                      arguments.train_dataset),
                                          logger=logger)
    explainer = Explainer(model=model, device=arguments.device,
                          train_dataset=arguments.train_dataset,
                          test_dataset=arguments.test_dataset,
                          local_exp_file_available=str2bool(arguments.local_exp_file_available),
                          loc_explain_mode=arguments.local_explainer,
                          aggregator=arguments.aggregator,
                          global_explainer=arguments.global_explainer,
                          max_skipgrams_length=arguments.max_skipgrams_length,
                          run_mode=arguments.run_mode,
                          logger=logger, out_folder=arguments.out_folder)
    start_time = time.time()
    explainer.run()
    end_time = time.time()
    logger.print_it('Explaining took: {}'.format(format_time(end_time - start_time)))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    options = get_options()
    explain(options)
