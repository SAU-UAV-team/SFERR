# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#buffer_herding+ chongjian yong

from copy import deepcopy
from typing import Tuple,List,Dict,Optional

import numpy as np
import torch
import torch.nn as nn

from utils.interpolator import Interpolator
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def icarl_replay(self, dataset, val_set_split=0):
    """
    Merge the replay buffer with the current task data.
    Optionally split the replay buffer into a validation set.

    :param self: the model instance
    :param dataset: the dataset
    :param val_set_split: the fraction of the replay buffer to be used as validation set
    """

    if self.task > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_train_mask = torch.zeros(len(dataset.train_loader.dataset.data)).bool()
        val_train_mask[torch.randperm(len(dataset.train_loader.dataset.data))[:buff_val_mask.sum()]] = True

        if val_set_split > 0:
            self.val_loader = deepcopy(dataset.train_loader)

        data_concatenate = torch.cat if isinstance(dataset.train_loader.dataset.data, torch.Tensor) else np.concatenate
        need_aug = hasattr(dataset.train_loader.dataset, 'not_aug_transform')
        if not need_aug:
            def refold_transform(x): return x.cpu()
        else:
            data_shape = len(dataset.train_loader.dataset.data[0].shape)
            if data_shape == 3:
                def refold_transform(x): return (x.cpu() * 255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)
            elif data_shape == 2:
                def refold_transform(x): return (x.cpu() * 255).squeeze(1).type(torch.uint8)

        # REDUCE AND MERGE TRAINING SET
        dataset.train_loader.dataset.targets = np.concatenate([
            dataset.train_loader.dataset.targets[~val_train_mask],
            self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
        ])
        dataset.train_loader.dataset.data = data_concatenate([
            dataset.train_loader.dataset.data[~val_train_mask],
            refold_transform((self.buffer.examples)[:len(self.buffer)][~buff_val_mask])
        ])

        if val_set_split > 0:
            # REDUCE AND MERGE VALIDATION SET
            self.val_loader.dataset.targets = np.concatenate([
                self.val_loader.dataset.targets[val_train_mask],
                self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
            ])
            self.val_loader.dataset.data = data_concatenate([
                self.val_loader.dataset.data[val_train_mask],
                refold_transform((self.buffer.examples)[:len(self.buffer)][buff_val_mask])
            ])


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:

    return num_seen_examples % buffer_portion_size + task * buffer_portion_size

class Buffer:
    def __init__(self, buffer_size, device, n_tasks=6, mode='herding', args=None):
        assert mode in ('ring', 'reservoir', 'herding')
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode) if mode != 'herding' else None
        self.mode = mode
        self.n_tasks = 6
        self.current_task = 0

        self.rebuilder = Interpolator(args, device) if args.model == "charon" else None
        self.buffer_size = buffer_size if self.rebuilder is None else int(
            buffer_size * self.rebuilder.sampling_interval)

        if mode == 'herding':
            self.class_prototypes = {}
            self.current_task_cache = {}
            self.protos_per_class = {}

        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks

        self.attributes = ['examples', 'labels', 'logits', 'task_labels', 'rebuilding_indexes', 'features','recon_losses']

    def set_current_task(self, task_id):
        self.current_task = task_id

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, features: torch.Tensor, task_labels: torch.Tensor,
                     rebuilding_indexes: torch.Tensor, **kwargs: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        """
        local_vars = {
            'examples': examples,
            'labels': labels,
            'logits': logits,
            'features': features,
            'task_labels': task_labels,
            'rebuilding_indexes': rebuilding_indexes
        }
        local_vars.update(kwargs)

        for attr_str in self.attributes:
            if attr_str in local_vars:
                attr = local_vars[attr_str]
            else:
                continue

            if attr is not None and not hasattr(self, attr_str):
                if attr_str == 'recon_losses':
                    typ = torch.float32
                else:
                    typ = torch.int64 if attr_str.endswith('els') or attr_str.endswith("xes") else torch.float32

                if hasattr(attr, 'shape'):
                    setattr(self, attr_str,
                            torch.zeros((self.buffer_size, *attr.shape[1:]), dtype=typ, device=self.device))
                else:
                    setattr(self, attr_str, torch.zeros(self.buffer_size, dtype=typ, device=self.device))

        for key, value in kwargs.items():
            if not hasattr(self, key):
                self.attributes.append(key)
                if key == 'recon_losses':
                    typ = torch.float32
                else:
                    typ = torch.int64 if key.endswith('els') or key.endswith("xes") else torch.float32

                if hasattr(value, 'shape'):
                    setattr(self, key, torch.zeros((self.buffer_size, *value.shape[1:]), dtype=typ, device=self.device))
                else:
                    setattr(self, key, torch.zeros(self.buffer_size, dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None, rebuilding_indexes=None,features=None, **kwargs):
        if self.rebuilder:
            examples, rebuilding_indexes = self.rebuilder.get_efficient_buffer_sample(examples)

        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, features,task_labels, rebuilding_indexes, **kwargs)
        if self.mode == 'herding':
            return self._cache_current_task_data(examples, labels, logits, task_labels,rebuilding_indexes, features, **kwargs)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].detach().to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].detach().to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].detach().to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].detach().to(self.device)
                if rebuilding_indexes is not None:
                    self.rebuilding_indexes[index] = rebuilding_indexes[i].detach().to(self.device)

                for key, value in kwargs.items():
                    if hasattr(self, key):
                        getattr(self, key)[index] = value[i].detach().to(self.device)

    def _cache_current_task_data(self, examples, labels, logits, task_labels, rebuilding_indexes, features, **kwargs):
        labels_np = labels.cpu().numpy()

        for i in range(examples.shape[0]):
            cls_label = int(labels_np[i])

            if cls_label not in self.current_task_cache:
                self.current_task_cache[cls_label] = {
                    'examples': [], 'features': [], 'labels': [],
                    'logits': [], 'rebuilding_indexes': [], 'recon_losses': []
                }

            self.current_task_cache[cls_label]['examples'].append(examples[i].cpu().clone())
            self.current_task_cache[cls_label]['features'].append(features[i])
            self.current_task_cache[cls_label]['labels'].append(labels[i].cpu().clone())

            if logits is not None:
                self.current_task_cache[cls_label]['logits'].append(logits[i].cpu().clone())
            if rebuilding_indexes is not None:
                self.current_task_cache[cls_label]['rebuilding_indexes'].append(
                    rebuilding_indexes[i].cpu().clone())
            if 'recon_losses' in kwargs and kwargs['recon_losses'] is not None:
                self.current_task_cache[cls_label]['recon_losses'].append(kwargs['recon_losses'][i].cpu().clone())

            n = examples.shape[0]
            assert labels.shape[0] == n
            if features is not None:
                assert features.shape[0] == n
            for key, value in kwargs.items():
                if key == 'recon_losses':
                    continue
                if key not in self.current_task_cache[cls_label]:
                    self.current_task_cache[cls_label][key] = []
                self.current_task_cache[cls_label][key].append(value[i].cpu().clone())


    def select_prototypes(self, current_task):
        if self.mode != 'herding' or not self.current_task_cache:
            return

        total_classes = len(set(list(self.class_prototypes.keys()) + list(self.current_task_cache.keys())))
        target_per_class = max(1, self.buffer_size // total_classes)
        new_per_class = (self.buffer_size - ((total_classes - 10) * target_per_class)) // 10

        print(f"Task {current_task}: Storing {new_per_class} samples per new class "
              f"({target_per_class} per old class) for {total_classes} total classes")

        class_thresholds = {}
        class_thresholds2 = {}
        for cls_label, cls_data in self.current_task_cache.items():
            if 'recon_losses' in cls_data and cls_data['recon_losses']:
                losses = [loss.item() for loss in cls_data['recon_losses']]
                class_thresholds[cls_label] = np.percentile(losses, 95)
                class_thresholds2[cls_label] = np.percentile(losses, 10)
            else:
                class_thresholds[cls_label] = float('inf')
                class_thresholds2[cls_label] = float('inf')
                print(f"Class {cls_label}: No reconstruction losses found, using all samples")

        for cls_label in list(self.class_prototypes.keys()):
            prototypes = self.class_prototypes[cls_label]
            if len(prototypes) > target_per_class:
                keep_indices = np.random.choice(len(prototypes), target_per_class, replace=False)
                self.class_prototypes[cls_label] = [prototypes[i] for i in keep_indices]

        for cls_label, cls_data in self.current_task_cache.items():
            if not cls_data.get('features') or not cls_data['features']:
                continue

            n_samples = len(cls_data['examples'])

            current_threshold = class_thresholds.get(cls_label, float('inf'))
            current_threshold2 = class_thresholds2.get(cls_label, float('inf'))
            filtered_indices = []
            filtered_features = []
            filtered_data = {
                'examples': [], 'features': [], 'labels': [],
                'logits': [], 'rebuilding_indexes': []
            }

            for i in range(n_samples):
                include_sample = True
                if 'recon_losses' in cls_data and i < len(cls_data['recon_losses']):
                    recon_loss = cls_data['recon_losses'][i].item()
                    if recon_loss > current_threshold or recon_loss < current_threshold2:
                        include_sample = False

                if include_sample:
                    filtered_indices.append(i)
                    filtered_features.append(cls_data['features'][i])
                    filtered_data['examples'].append(cls_data['examples'][i])
                    filtered_data['features'].append(cls_data['features'][i])
                    filtered_data['labels'].append(cls_data['labels'][i])
                    if 'logits' in cls_data and i < len(cls_data['logits']):
                        filtered_data['logits'].append(cls_data['logits'][i])
                    if 'rebuilding_indexes' in cls_data and i < len(cls_data['rebuilding_indexes']):
                        filtered_data['rebuilding_indexes'].append(cls_data['rebuilding_indexes'][i])

            print(
                f"Class {cls_label}: {len(filtered_indices)}/{n_samples} samples passed reconstruction difficulty filter "
                f"(threshold: {current_threshold:.4f})")

            if len(filtered_indices) < new_per_class:
                print(f"Warning: Too few easy samples for class {cls_label}, using all samples")
                filtered_data = {k: v for k, v in cls_data.items() if k != 'recon_losses'}
                filtered_features = cls_data['features']

            num_to_select = min(new_per_class, len(filtered_data['examples']))

            features = torch.stack(filtered_features).cpu().numpy()
            features = features.reshape(features.shape[0], -1)
            selected_idxs = self.herding_algorithm_with_target(
                features=features,
                nb_protos=num_to_select,
            )
            print(f"Selected {len(selected_idxs)} prototypes for class {cls_label}")
            self.class_prototypes[cls_label] = []
            for idx in selected_idxs:
                prototype = {
                    'example': filtered_data['examples'][idx].to(self.device),
                    'label': filtered_data['labels'][idx].to(self.device),
                    'feature': filtered_data['features'][idx].to(self.device),
                }

                if 'logits' in filtered_data and idx < len(filtered_data['logits']):
                    prototype['logit'] = filtered_data['logits'][idx].to(self.device)
                if 'rebuilding_indexes' in filtered_data and idx < len(filtered_data['rebuilding_indexes']):
                    prototype['rebuilding_index'] = filtered_data['rebuilding_indexes'][idx].to(self.device)

                self.class_prototypes[cls_label].append(prototype)
        self._update_buffer_from_prototypes()
        self._balance_buffer()
        for cls_data in self.current_task_cache.values():
            if 'recon_losses' in cls_data:
                del cls_data['recon_losses']


    def _balance_buffer(self):
        total_samples = sum(len(protos) for protos in self.class_prototypes.values())
        while total_samples < self.buffer_size:
            min_class = min(self.class_prototypes.keys(),
                            key=lambda cls: len(self.class_prototypes[cls]))
            if self.current_task_cache.get(min_class):
                new_sample = self.current_task_cache[min_class].pop(0)
                self.class_prototypes[min_class].append({
                    'example': new_sample['examples'][0],
                    'label': new_sample['labels'][0],
                    'feature': new_sample['features'][0].cpu().clone()
                })
                total_samples += 1
            else:
                cls = list(self.class_prototypes.keys())[0]
                proto = self.class_prototypes[cls][0]
                self.class_prototypes[cls].append(deepcopy(proto))
                total_samples += 1

        self._update_buffer_from_prototypes()


    def _update_buffer_from_prototypes(self):
        all_examples, all_labels, all_logits,all_features  = [], [], [],[]
        all_rebuilding_indexes = []

        sorted_classes = sorted(self.class_prototypes.keys())
        for cls in sorted_classes:
            for proto in self.class_prototypes[cls]:
                all_examples.append(proto['example'])
                all_labels.append(proto['label'])
                all_features.append(proto['feature'])

                if 'logit' in proto:
                    all_logits.append(proto['logit'])
                if 'rebuilding_index' in proto:
                    all_rebuilding_indexes.append(proto['rebuilding_index'])

        examples_tensor = torch.stack(all_examples)
        labels_tensor = torch.stack(all_labels)
        features_tensor = torch.stack(all_features)
        logits_tensor = torch.stack(all_logits) if all_logits else None
        rebuilding_tensor = torch.stack(all_rebuilding_indexes) if all_rebuilding_indexes else None
        num_prototypes = len(all_examples)

        if not hasattr(self, 'examples'):
            self.init_tensors(examples_tensor, labels_tensor, logits_tensor,features = features_tensor,
                              task_labels=None, rebuilding_indexes=rebuilding_tensor)
        if num_prototypes > self.buffer_size:
            keep_indices = torch.randperm(num_prototypes)[:self.buffer_size]
            examples_tensor = examples_tensor[keep_indices]
            labels_tensor = labels_tensor[keep_indices]
            features_tensor = features_tensor[keep_indices]
            if logits_tensor is not None:
                logits_tensor = logits_tensor[keep_indices]
            if rebuilding_tensor is not None:
                rebuilding_tensor = rebuilding_tensor[keep_indices]
            num_prototypes = self.buffer_size

        self.examples[:num_prototypes] = examples_tensor
        self.labels[:num_prototypes] = labels_tensor
        self.features[:num_prototypes] = features_tensor

        if logits_tensor is not None:
            self.logits[:num_prototypes] = logits_tensor
        if rebuilding_tensor is not None and hasattr(self, 'rebuilding_indexes'):
            self.rebuilding_indexes[:num_prototypes] = rebuilding_tensor

        self.num_seen_examples = num_prototypes

    @staticmethod
    def herding_algorithm_with_target(features: np.ndarray, nb_protos: int) -> np.ndarray:

        if features.size == 0 or nb_protos <= 0:
            return np.array([])
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        class_mean = np.mean(features, axis=0)
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
        mu = class_mean / (np.linalg.norm(class_mean) + 1e-10)
        w_t = mu.copy()
        selected_indices = []
        available_indices = np.arange(features.shape[0])

        for _ in range(min(nb_protos, features.shape[0])):
            similarities = np.dot(features_norm[available_indices], w_t)
            best_local_idx = np.argmax(similarities)
            best_idx = available_indices[best_local_idx]
            selected_indices.append(best_idx)
            w_t += mu - features_norm[best_idx]
            available_indices = np.delete(available_indices, best_local_idx)
            if len(available_indices) == 0:
                print('available_indices == 0')
                break

        return np.array(selected_indices)


    def get_data(self, size: int, transform: nn.Module = None, return_index=False) -> Tuple:

        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None:
            def transform(x): return x
        # TODO Possible thing to fasten more is to avoid transform in cpu and get back to cuda immediately after
        ret_tuple = [torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device), ]
        ret_tuple_content = ["examples", ]
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += [attr[choice], ]
                ret_tuple_content += [attr_str]

        if self.rebuilder:
            ret_tuple[0] = self.rebuilder(ret_tuple[0], ret_tuple[ret_tuple_content.index('rebuilding_indexes')])
            ret_tuple.pop()

        if not return_index:
            return tuple(ret_tuple)
        else:
            return (torch.tensor(choice).to(self.device), ) + tuple(ret_tuple)

    def get_data_by_index(self, indexes, transform: nn.Module = None) -> Tuple:

        if transform is None:
            def transform(x): return x
        ret_tuple = [torch.stack([transform(ee.cpu()) for ee in self.examples[indexes]]).to(self.device),]
        ret_tuple_content = ["examples", ]
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += [attr[indexes], ]
                ret_tuple_content += [attr_str]

        if self.rebuilder:
            # print(f"data shape before rebuild:{ret_tuple[0].shape}")
            ret_tuple[0] = self.rebuilder(ret_tuple[0], ret_tuple[ret_tuple_content.index('rebuilding_indexes')])
            # print(f"data shape after rebuild:{ret_tuple[0].shape}")
            ret_tuple.pop()
            # print(f"data shape after rebuild:{ret_tuple[0].shape}")

        return ret_tuple

    def is_empty(self) -> bool:

        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: nn.Module = None) -> Tuple:

        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:

        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0