import os
import shutil
import fire
import yaml
import torch
import csv
import random
import logging

from typing import Dict, Optional, List, Tuple
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from model.supernet import SupernetClassifier, SupernetClassifierSpec

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss


def create_model(model_spec_dict: Dict, device: torch.device) -> SupernetClassifier:

    model_spec = SupernetClassifierSpec.from_dict(model_spec_dict)
    model = SupernetClassifier.from_spec(model_spec)
    model.to(device)

    return model


def create_dataloader(batch_size: int, train: bool) -> DataLoader:

    dataset = MNIST(
        download=True,
        root=".",
        train=train,
        transform=Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train
    )

def create_trainer(
    model,
    available_model_configurations,
    seed,
    train_dl,
    val_dl,
    device,
    lr,
    early_stopping_patience,
    output_dir,
    experiment_name,
    report_fields
):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss = torch.nn.NLLLoss()

    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(
        model,
        metrics={
            'accuracy': Accuracy(),
            'nll': Loss(loss)
        },
        device=device
    )

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names='all')

    @trainer.on(Events.ITERATION_STARTED)
    def set_model_configuration(engine):
        configuration = random.sample(available_model_configurations, 1)[0]
        model.reconfigure(configuration)

    report_filename = os.path.join(output_dir, 'report.csv')
    if not os.path.exists(report_filename):
        with open(report_filename, 'w') as report_file:
            writer = csv.DictWriter(report_file, fieldnames=report_fields)
            writer.writeheader()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):

        pbar.n = pbar.last_print_n = 0

        evaluator.run(train_dl)
        train_metrics = evaluator.state.metrics
        pbar.log_message(
            'Train Results - Epoch: {}\n\t{}'.format(
                engine.state.epoch,
                '\n\t'.join([
                    f'train_{name}:\t\t{value}'
                    for name, value in train_metrics.items()
                ])
            )
        )

        evaluator.run(val_dl)
        val_metrics = evaluator.state.metrics
        pbar.log_message(
            'Validation Results - Epoch: {}\n\t{}'.format(
                engine.state.epoch,
                '\n\t'.join([
                    f'val_{name}:\t\t{value}'
                    for name, value in val_metrics.items()
                ])
            )
        )

        with open(report_filename, 'a') as report_file:
            writer = csv.DictWriter(report_file, fieldnames=report_fields)
            writer.writerow({
                'experiment': experiment_name,
                'epoch': engine.state.epoch,
                'val_accuracy': val_metrics['accuracy'],
                'train_accuracy': train_metrics['accuracy']
            })

    def get_accuracy(engine):
        return engine.state.metrics['accuracy']

    checkpoint_handler = ModelCheckpoint(
        dirname=output_dir,
        filename_prefix=experiment_name,
        n_saved=1,
        score_function=get_accuracy,
        require_empty=False
    )

    evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {'model': model})

    early_stopping_handler = EarlyStopping(
        patience=early_stopping_patience,
        score_function=get_accuracy,
        trainer=trainer
    )

    evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

    return trainer


def train(experiment_name: str = 'baseline',
          config_path: str = '../config.yml',
          output_dir: str = '../results',
          report_fields: List[str] = ('experiment', 'epoch', 'val_accuracy', 'train_accuracy'),
          model_configuration: Optional[Tuple[str]] = None,
          seed: Optional[int] = 42,
          logging_level: str = 'INFO'):

    logging.basicConfig(level=logging_level)

    with open(config_path) as config_file:
        config = yaml.load(config_file, yaml.SafeLoader)

    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(output_dir, f'config_{experiment_name}.yml'))

    logging.info(f'Setting seed: {seed}')
    torch.manual_seed(seed)

    train_dataloader = create_dataloader(batch_size=config['batch_size'], train=True)
    val_dataloader = create_dataloader(batch_size=config['batch_size'], train=False)

    device = torch.device(config['device'])
    model = create_model(config['model_spec'], device=device)
    logging.info('Model loaded:\n%s', model)

    available_model_configurations = set(model.get_available_configurations())\
        if model_configuration is None \
        else {model_configuration}
    logging.info('Using model configurations:\n%s', available_model_configurations)

    trainer = create_trainer(
        model,
        available_model_configurations=available_model_configurations,
        seed=seed,
        train_dl=train_dataloader,
        val_dl=val_dataloader,
        device=device,
        lr=config['lr'],
        early_stopping_patience=config['early_stopping_patience'],
        output_dir=output_dir,
        experiment_name=experiment_name,
        report_fields=report_fields
    )
    trainer.run(train_dataloader, max_epochs=config['num_epoch'])


if __name__ == '__main__':
    fire.Fire(train)
