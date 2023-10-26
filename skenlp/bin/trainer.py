import os
import time
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup
from scipy.special import expit
from typing import Union

from skenlp.data import format_dataset_name, get_dataset, check_train_test_datasets
from skenlp.utils import format_time, select_device, print_message, DumbLogger, SmartLogger
from skenlp.metrics import Accuracy, F1


class Trainer:
    def __init__(self,
                 model,
                 device,
                 train_dataset: str = 'sst',
                 test_dataset: str = 'sst',
                 optim: str = 'adamw',
                 epochs: int = 10,
                 lr: float = 5e-5,
                 batch_size: int = 16,
                 dropout: float = 0.1,
                 threshold: float = 0,
                 metrics: list[str] = None,
                 logger: Union[DumbLogger, SmartLogger] = None,
                 out_folder: str = 'outs'):
        # self.config = config
        if metrics is None:
            metrics = []
        self.epoch = None
        self.model = model
        self.logger = logger if logger is not None else DumbLogger()
        self.device = select_device(device=device,
                                    batch_size=batch_size,
                                    logger=self.logger)

        self.model.to(self.device)

        if not check_train_test_datasets(train_dataset, test_dataset):
            raise ValueError('Datasets \"{}\" and \"{}\" are not compatible for train and test!'.format(train_dataset,
                                                                                                        test_dataset))
        self.train_dataset_config = format_dataset_name(train_dataset)
        self.test_dataset_config = format_dataset_name(test_dataset)
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.train_dataset = get_dataset(self.train_dataset_config,
                                         tokenizer=self.tokenizer,
                                         split='train',
                                         logger=self.logger)
        self.test_dataset = get_dataset(self.test_dataset_config,
                                        tokenizer=self.tokenizer,
                                        split='test',
                                        logger=self.logger)

        if self.train_dataset_config['name'] in ['sst', 'imdb', 'agnews', 'sms', 'trec', 'youtube']:
            self.loss = CrossEntropyLoss()
        elif self.train_dataset_config['name'] in ['moral']:
            self.loss = BCEWithLogitsLoss()
            self.pred_threshold = 0.
        else:
            raise ValueError('Invalid dataset selection with dataset {}'.format(self.train_dataset_config['name']))

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout

        optim_options = {'adamw': AdamW, }
        self.optim = optim_options[optim](self.model.parameters(),
                                          lr=self.lr)

        # Setup metrics depending on the choice
        self.metrics = {}
        for metric in metrics:
            if metric == 'acc':
                self.metrics[metric] = Accuracy(dataset=self.train_dataset_config['name'])
            elif metric == 'f1':
                self.metrics[metric] = F1(dataset=self.train_dataset_config['name'])
            else:
                raise ValueError('The metric {} is not available in classification mode!'.format(metric))

        # Setup checkpoint folder for the best model to be found
        self.checkpoint_dir = os.path.join(out_folder, 'models')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train(self, validation: bool = True):
        # Move datasets to device
        self.train_dataset.to(self.device)
        self.test_dataset.to(self.device)
        # Create loaders from datasets
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        dev_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)

        # validation_step = math.floor(len(train_loader) / 4)
        # Define initial best loss to infinity
        best_loss = math.inf
        # Keep track of time
        epoch_t0 = time.time()
        # Define learning rate scheduler for the optimiser
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(self.optim, num_warmup_steps=0, num_training_steps=total_steps)
        # Set model to training mode
        self.model.train()
        # Log start of training
        self.logger.print_it("Training...")
        # Loop over epochs
        for epoch in range(self.epochs):
            self.epoch = epoch
            running_loss = 0.
            running_scores = {met_name: 0.0 for met_name in self.metrics.keys()}
            epoch_starting_time = time.time()
            # Iterate over batches of the training loader
            for batch_index, batch in enumerate(train_loader):
                # Zero gradient of optimizer
                self.optim.zero_grad()
                # Input batch to the model
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                if self.train_dataset_config['name'] in ['sst', 'imdb', 'agnews', 'sms', 'trec', 'youtube']:
                    labels = batch['labels']
                elif self.train_dataset_config['name'] in ['moral']:
                    labels = batch['labels'].float()
                else:
                    raise ValueError('Something wrong with dataset labels in training...')
                output = self.model(input_ids, attention_mask=attention_mask)
                # Get loss and compute gradient
                loss = self.loss(output, labels)
                loss.backward()
                # Clip gradient to avoid gradient explosion issues
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # Update model
                self.optim.step()
                scheduler.step()
                # Print message
                running_loss += loss
                # Compute performance metrics
                if self.train_dataset_config['name'] in ['moral']:
                    output = (output > self.pred_threshold).float()
                for metric_name, metric_obj in self.metrics.items():
                    running_scores[metric_name] += metric_obj.compute(y_pred=output,
                                                                      y_true=labels)
                # Average out loss and metrics
                avg_loss = running_loss / (batch_index + 1)
                avg_scores = {met_name: met_value / (batch_index + 1) for met_name, met_value in
                              running_scores.items()}
                print_message(logger=self.logger,
                              epoch=self.epoch + 1,
                              tot_epochs=self.epochs,
                              index=batch_index + 1,
                              size=len(train_loader),
                              loss=avg_loss,
                              metrics=avg_scores,
                              time=time.time() - epoch_starting_time,
                              mode='train')
            self.logger.set_logger_newline()
            # Run validation metrics over the dev dataset if required
            if validation:
                y_predicted, y_true, loss = self.evaluate(dev_loader)
                if loss < best_loss:
                    best_loss = loss
                    self.save_best_model()
                self.save_last_model()
        self.logger.print_it('Total training time: {}'.format(format_time(time.time() - epoch_t0)))

    def save_best_model(self):
        checkpoint = os.path.join(self.checkpoint_dir, 'best_{}_{}.pt'.format(self.model.name,
                                                                              self.train_dataset_config['full']))
        self.logger.print_it('Found best model up to now. Storing it to {}...'.format(checkpoint))
        torch.save(self.model.state_dict(), checkpoint)

    def save_last_model(self):
        checkpoint = os.path.join(self.checkpoint_dir, 'last_{}_{}.pt'.format(self.model.name,
                                                                              self.train_dataset_config['full']))
        self.logger.print_it('Storing last checkpoint to {}...'.format(checkpoint))
        torch.save(self.model.state_dict(), checkpoint)

    def evaluate(self, loader):
        self.model.eval()
        y_predicted = []
        y_true = []
        total_loss = 0
        running_scores = {met_name: 0.0 for met_name in self.metrics.keys()}
        evaluate_starting_time = time.time()
        for batch_index, batch in enumerate(loader):
            with torch.no_grad():
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                if self.test_dataset_config['name'] in ['sst', 'imdb', 'agnews', 'sms', 'trec', 'youtube']:
                    labels = batch['labels']
                elif self.test_dataset_config['name'] in ['moral']:
                    labels = batch['labels'].float()
                else:
                    raise ValueError('Something wrong with dataset labels in evaluate...')
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
            if self.test_dataset_config['name'] in ['sst', 'imdb', 'agnews', 'sms', 'trec', 'youtube']:
                logits = outputs
            elif self.test_dataset_config['name'] in ['moral']:
                logits = (outputs > self.pred_threshold).float()
            else:
                raise ValueError('Something wrong with dataset labels in evaluate...')
            # Compute performance metrics
            for metric_name, metric_obj in self.metrics.items():
                running_scores[metric_name] += metric_obj.compute(y_pred=logits,
                                                                  y_true=labels)
            # Average out loss and metrics
            avg_scores = {met_name: met_value / (batch_index + 1) for met_name, met_value in
                          running_scores.items()}
            avg_loss = total_loss / (batch_index + 1)
            print_message(logger=self.logger,
                          epoch=self.epoch + 1,
                          tot_epochs=self.epochs,
                          index=batch_index + 1,
                          size=len(loader),
                          loss=avg_loss,
                          metrics=avg_scores,
                          time=time.time() - evaluate_starting_time,
                          mode='eval')
            true_values = batch['labels']
            y_predicted.extend(logits)
            y_true.extend(true_values)
        self.model.train()
        self.logger.set_logger_newline()
        return torch.stack(y_predicted), torch.stack(y_true), total_loss / len(loader)

    def predict(self, dataset, save=False):
        dataset.to(self.device)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        y_predicted, y_true, loss = self.evaluate(loader)
        self.logger.print_it(f'Prediction loss: {loss}')
        if save:
            np.savetxt('predicted.txt', expit(y_predicted.cpu().numpy()), fmt='%1.2f')
            np.savetxt('true.txt', y_true.cpu().numpy(), fmt='%1d')
        predictions = (y_predicted > self.threshold).float().cpu().numpy()
        return predictions, y_true.cpu().numpy()
