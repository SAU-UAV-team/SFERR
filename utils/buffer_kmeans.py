from copy import deepcopy
from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

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
    随机填策略
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    工作原理：当缓冲区未满时，新样本直接存入；缓冲区满后，新样本以buffer_size/num_seen_examples的概率随机替换旧样本
    结果：最终缓冲区包含早期样本的概率为buffer_size/总样本数，并非全部保留
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    '''
    工作原理：将缓冲区均分给每个任务，每个子缓冲区独立进行先进先出替换
    结果：每个任务最多保留buffer_size/n_tasks个样本，旧任务的样本会被新任务数据覆盖
    '''
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    def __init__(self, buffer_size, device, n_tasks=6, mode='kmeans', args=None):
        assert mode in ('ring', 'reservoir', 'kmeans')
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode) if mode != 'kmeans' else None
        self.mode = mode
        self.n_tasks = 6
        self.current_task = 0

        self.rebuilder = Interpolator(args, device) if args and args.model == "charon" else None
        self.buffer_size = buffer_size if self.rebuilder is None else int(
            buffer_size * self.rebuilder.sampling_interval)

        if mode == 'kmeans':
            self.class_prototypes = {}
            self.current_task_cache = {}
            self.protos_per_class = {}
            self.n_strata = 5

        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks

        self.attributes = ['examples', 'labels', 'logits', 'task_labels', 'rebuilding_indexes', 'features']

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
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') or attr_str.endswith("xes") else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size, *attr.shape[1:]), dtype=typ, device=self.device))

        for key, value in kwargs.items():
            if not hasattr(self, key):
                self.attributes.append(key)
                typ = torch.int64 if key.endswith('els') or key.endswith("xes") else torch.float32
                setattr(self, key, torch.zeros((self.buffer_size, *value.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None, rebuilding_indexes=None, features=None, **kwargs):
        if self.rebuilder:
            examples, rebuilding_indexes = self.rebuilder.get_efficient_buffer_sample(examples)

        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, features, task_labels, rebuilding_indexes, **kwargs)

        # 距离分层采样模式：先缓存当前任务数据
        if self.mode == 'kmeans':
            return self._cache_current_task_data(examples, labels, logits, task_labels, rebuilding_indexes, features, **kwargs)

        # 原有模式（reservoir/ring）
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
                if features is not None:
                    self.features[index] = features[i].detach().to(self.device)
                for key, value in kwargs.items():
                    if hasattr(self, key):
                        getattr(self, key)[index] = value[i].detach().to(self.device)

    def _cache_current_task_data(self, examples, labels, logits, task_labels, rebuilding_indexes, features, **kwargs):
        """为距离分层采样缓存当前任务数据（按类别组织）"""
        labels_np = labels.cpu().numpy()
        for i in range(examples.shape[0]):
            cls_label = int(labels_np[i])
            if cls_label not in self.current_task_cache:
                self.current_task_cache[cls_label] = {
                    'examples': [], 'features': [], 'labels': [],
                    'logits': [], 'rebuilding_indexes': []
                }
            self.current_task_cache[cls_label]['examples'].append(examples[i].cpu().clone())
            self.current_task_cache[cls_label]['features'].append(features[i].cpu().clone())
            self.current_task_cache[cls_label]['labels'].append(labels[i].cpu().clone())
            if logits is not None:
                self.current_task_cache[cls_label]['logits'].append(logits[i].cpu().clone())
            if rebuilding_indexes is not None:
                self.current_task_cache[cls_label]['rebuilding_indexes'].append(
                    rebuilding_indexes[i].cpu().clone()
                )
            for key, value in kwargs.items():
                if key not in self.current_task_cache[cls_label]:
                    self.current_task_cache[cls_label][key] = []
                self.current_task_cache[cls_label][key].append(value[i].cpu().clone())

    def select_prototypes(self, current_task):
        """使用 K-Means 聚类选择每个类别的代表性原型样本"""
        if self.mode != 'kmeans' or not self.current_task_cache:
            return

        total_classes = len(set(list(self.class_prototypes.keys()) + list(self.current_task_cache.keys())))
        target_per_class = max(1, self.buffer_size // total_classes)
        new_per_class = target_per_class + 1 if self.buffer_size % total_classes > 0 else target_per_class

        print(f"Task {self.current_task}: Storing {new_per_class} samples per new class "
              f"({target_per_class} per old class) for {total_classes} total classes")

        # 1. 对旧类别进行 K-Means 重采样（保持每类 target_per_class 个）
        for cls_label in list(self.class_prototypes.keys()):
            prototypes = self.class_prototypes[cls_label]
            if len(prototypes) > target_per_class:
                features_list = [p['feature'].cpu().numpy() for p in prototypes]
                features_matrix = np.vstack(features_list)
                selected_idxs = self._kmeans_sampling(features_matrix, target_per_class)
                self.class_prototypes[cls_label] = [prototypes[i] for i in selected_idxs]

        # 2. 为新类别选择样本
        for cls_label, cls_data in self.current_task_cache.items():
            if not cls_data.get('features') or not cls_data['features']:
                continue

            n_samples = len(cls_data['examples'])
            num_to_select = min(new_per_class, n_samples)

            features = torch.stack(cls_data['features']).cpu().numpy()
            features = features.reshape(features.shape[0], -1)

            selected_idxs = self._kmeans_sampling(features, num_to_select)

            print(f"Selected {len(selected_idxs)} prototypes for class {cls_label} (from {n_samples} samples)")

            self.class_prototypes[cls_label] = []
            for idx in selected_idxs:
                prototype = {
                    'example': cls_data['examples'][idx].to(self.device),
                    'label': cls_data['labels'][idx].to(self.device),
                    'feature': cls_data['features'][idx].to(self.device),
                }
                if 'logits' in cls_data and idx < len(cls_data['logits']):
                    prototype['logit'] = cls_data['logits'][idx].to(self.device)
                if 'rebuilding_indexes' in cls_data and idx < len(cls_data['rebuilding_indexes']):
                    prototype['rebuilding_index'] = cls_data['rebuilding_indexes'][idx].to(self.device)
                self.class_prototypes[cls_label].append(prototype)

        # 3. 更新缓冲区
        self._update_buffer_from_prototypes()

        # 4. 平衡缓冲区（如果仍不足，则补充样本）
        self._balance_buffer()

    def _kmeans_sampling(self, features: np.ndarray, n_samples: int) -> np.ndarray:
        """
        使用 K-Means 聚类选择代表性样本。
        :param features: 样本特征数组 (n, d)
        :param n_samples: 需要的样本数量（即簇数）
        :return: 选中的样本索引（一维数组）
        """
        if features.size == 0 or n_samples <= 0:
            return np.array([])

        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        n_total = features.shape[0]
        if n_samples >= n_total:
            return np.arange(n_total)

        kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
        kmeans.fit(features)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        selected_indices = []
        for cluster_id in range(n_samples):
            cluster_mask = (labels == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
                remaining = list(set(range(n_total)) - set(selected_indices))
                if remaining:
                    selected_indices.append(np.random.choice(remaining))
                continue
            center = centers[cluster_id]
            distances = np.linalg.norm(features[cluster_indices] - center, axis=1)
            nearest_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(nearest_idx)

        if len(selected_indices) < n_samples:
            remaining = list(set(range(n_total)) - set(selected_indices))
            if remaining:
                additional = np.random.choice(remaining, size=n_samples - len(selected_indices), replace=False)
                selected_indices.extend(additional.tolist())

        return np.array(selected_indices[:n_samples])

    def _balance_buffer(self):
        total_samples = sum(len(protos) for protos in self.class_prototypes.values())

        while total_samples < self.buffer_size:
            min_class = min(self.class_prototypes.keys(),
                            key=lambda cls: len(self.class_prototypes[cls]))

            if self.current_task_cache.get(min_class):
                # 添加一个新样本
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
        all_examples, all_labels, all_logits, all_features = [], [], [], []
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
            self.init_tensors(examples_tensor, labels_tensor, logits_tensor,
                              features=features_tensor, task_labels=None,
                              rebuilding_indexes=rebuilding_tensor)

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

    def get_data(self, size: int, transform: nn.Module = None, return_index=False) -> Tuple:
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None:
            def transform(x): return x

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
            ret_tuple[0] = self.rebuilder(ret_tuple[0], ret_tuple[ret_tuple_content.index('rebuilding_indexes')])
            ret_tuple.pop()

        return ret_tuple

    def is_empty(self) -> bool:
        return self.num_seen_examples == 0

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
