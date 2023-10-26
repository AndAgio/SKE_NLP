import time
import warnings

from skenlp.models import BertConfig, BertForSequenceClassification
from skenlp.data import get_label_names
from skenlp.bin import Trainer
from skenlp.utils import set_seeds, get_logger, get_options


def train(arguments):
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    set_seeds(seed_val)
    logger = get_logger(arguments=arguments, mode='smart')

    logger.print_it('Training {} over {} and testing it over {}'.format(arguments.model.upper(),
                                                                        arguments.train_dataset.upper(),
                                                                        arguments.test_dataset.upper()))
    model = BertForSequenceClassification(BertConfig(),
                                          num_labels=len(get_label_names(arguments.train_dataset)),
                                          logger=logger)
    trainer = Trainer(model=model,
                      device=arguments.device,
                      train_dataset=arguments.train_dataset,
                      test_dataset=arguments.test_dataset,
                      optim=arguments.optim,
                      epochs=arguments.epochs,
                      lr=arguments.lr,
                      batch_size=arguments.batch_size,
                      dropout=arguments.dropout,
                      threshold=arguments.threshold,
                      metrics=arguments.metrics,
                      logger=logger,
                      out_folder=arguments.out_folder, )
    start_time = time.time()
    trainer.train(validation=True)
    end_time = time.time()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    options = get_options()
    train(options)
