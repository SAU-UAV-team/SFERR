# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
from argparse import Namespace
from typing import Tuple
import time

import torch
from tqdm import tqdm

from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.charon import CHARON
from models.utils.continual_model import ContinualModel
from utils.loggers import *
from utils.status import ProgressBar

try:
    import wandb
except ImportError:
    wandb = None


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
            dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task, and per-class accuracy
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes, per_class_accs = [], [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0

        class_correct = [0.0] * dataset.N_CLASSES_PER_TASK
        class_total = [0.0] * dataset.N_CLASSES_PER_TASK
        class_correct_mask = [0.0] * dataset.N_CLASSES_PER_TASK

        for data in tqdm(test_loader, desc=f'Evaluation of task {k}'):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.float().to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                for label, prediction in zip(labels, pred):
                    class_idx = label.item() % dataset.N_CLASSES_PER_TASK
                    if label == prediction:
                        class_correct[class_idx] += 1
                    class_total[class_idx] += 1

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred_mask = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred_mask == labels).item()

                    for label, prediction in zip(labels, pred_mask):
                        class_idx = label.item() % dataset.N_CLASSES_PER_TASK
                        if label == prediction:
                            class_correct_mask[class_idx] += 1

        class_accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                            for i in range(dataset.N_CLASSES_PER_TASK)]

        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
        per_class_accs.append(class_accuracies)

        print(f"\nTask {k} Per-Class Accuracy:")
        for i, acc in enumerate(class_accuracies):
            global_class_idx = k * dataset.N_CLASSES_PER_TASK + i
            print(f"Class {global_class_idx} (Task-local {i}): {acc:.2f}%")

    model.net.train(status)
    return accs, accs_mask_classes


def save_task_checkpoint(model: ContinualModel, dataset: ContinualDataset,
                         args: Namespace, task_idx: int, task_accs: list = None) -> str:

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"./saved_models/{model.NAME}_{dataset.NAME}"
    os.makedirs(save_dir, exist_ok=True)

    filename = f"task_{task_idx}_{timestamp}.pth"
    filepath = os.path.join(save_dir, filename)

    checkpoint = {
        'model_state_dict': model.net.state_dict(),
        'model_name': model.NAME,
        'dataset_name': dataset.NAME,
        'task_idx': task_idx,
        'timestamp': timestamp,
        'args': vars(args),
        'training_completed': False,
    }

    if task_accs is not None:
        checkpoint['task_accuracies'] = task_accs

    torch.save(checkpoint, filepath)
    print(f"\n{'=' * 50}")
    print(f"Task {task_idx} checkpoint saved to: {filepath}")
    print(f"{'=' * 50}")

    return filepath


def save_final_model(model: ContinualModel, dataset: ContinualDataset, args: Namespace) -> str:

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"./saved_models/{model.NAME}_{dataset.NAME}"
    os.makedirs(save_dir, exist_ok=True)

    filename = f"final_{timestamp}.pth"
    filepath = os.path.join(save_dir, filename)

    checkpoint = {
        'model_state_dict': model.net.state_dict(),
        'model_name': model.NAME,
        'dataset_name': dataset.NAME,
        'timestamp': timestamp,
        'args': vars(args),
        'training_completed': True,
    }

    torch.save(checkpoint, filepath)
    print(f"\n{'=' * 50}")
    print(f"Final model saved to: {filepath}")
    print(f"{'=' * 50}")

    return filepath



def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)

    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        name = f'{model.NAME}_{dataset.NAME}_bs{args.batch_size}_lr{args.lr}_ep{args.n_epochs}'
        name += f'_{args.rebuilder}' if hasattr(args, 'rebuilder') else ''
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=name)
        args.wandb_url = wandb.run.get_url()

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)


    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, _ = dataset.get_data_loaders()

        ###############################################
        task_samples = len(train_loader.dataset)
        print(f"\n{'=' * 50}")
        print(f"Starting Task {t + 1}/{dataset.N_TASKS}")
        print(f"Task Samples: {task_samples}")
        print(f"Buffer Size: {model.buffer.buffer_size}")
        print(f"Buffer mode: {model.buffer.mode}")
        print(f"Total Seen Examples: {model.buffer.num_seen_examples}")
        print(f"Current Buffer Usage: {len(model.buffer)}/{model.buffer.buffer_size}")
        print(f"{'=' * 50}\n")
        ###############################################

        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics and args.model != 'joint':
            accs = evaluate(model, dataset, last=True)
            results[t - 1] = results[t - 1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)
        for epoch in range(model.args.n_epochs):
            if args.model == 'joint':
                continue

            if hasattr(model, 'begin_epoch'):
                model.begin_epoch(epoch)

            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break

                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.float().to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.float().to(model.device)
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)

                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.float().to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.float().to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs)

                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

            if scheduler is not None:
                scheduler.step()

            if hasattr(model, 'end_epoch'):
                model.end_epoch(epoch)

        if hasattr(model, 'end_task'):
            if isinstance(model, CHARON):
                model.end_task(train_loader)
            else:
                model.end_task(dataset)

        if args.model == 'joint' and t < dataset.N_TASKS - 1:
            continue

        accs = evaluate(model, dataset)
        print(f"acc1{accs}")
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        save_task_checkpoint(model, dataset, args, t, accs)

        if not args.disable_log:
            logger.log(mean_acc)
            logger.log_fullacc(accs)

        if not args.nowand:
            d2 = {'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                  **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                  **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])},
                  'RESULT_task': t}

            wandb.log(d2)

    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                           results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()