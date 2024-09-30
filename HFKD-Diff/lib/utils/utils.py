import time
from datetime import datetime

import numpy as np
from timm.data import ImageDataset

from torchvision.datasets import CIFAR100


class ImageNetInstanceSample(ImageDataset):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self, root, name, class_map, load_bytes, is_sample=False, k=4096, **kwargs):
        super().__init__(root, parser=name, class_map=class_map, load_bytes=load_bytes, **kwargs)
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 1000
            num_samples = len(self.parser)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                _, target = self.parser[i]
                label[i] = target

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = super().__getitem__(index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


class CIFAR100InstanceSample(CIFAR100, ImageNetInstanceSample):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self, root, train, is_sample=False, k=4096, **kwargs):
        CIFAR100.__init__(self, root, train, **kwargs)
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 100
            num_samples = len(self.data)

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[self.targets[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        img, target = CIFAR100.__getitem__(self, index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


class TimePredictor:
    def __init__(self, steps, most_recent=30, drop_first=True):
        self.init_time = time.time()
        self.steps = steps
        self.most_recent = most_recent
        self.drop_first = drop_first  # drop iter 0

        self.time_list = []
        self.temp_time = self.init_time

    def update(self):
        time_interval = time.time() - self.temp_time
        self.time_list.append(time_interval)

        if self.drop_first and len(self.time_list) > 1:
            self.time_list = self.time_list[1:]
            self.drop_first = False

        self.time_list = self.time_list[-self.most_recent:]
        self.temp_time = time.time()

    def get_pred_text(self):
        single_step_time = np.mean(self.time_list)
        end_timestamp = self.init_time + single_step_time * self.steps
        return datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    
import logging
DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)


_logger = logging.getLogger(__name__)


def resolve_data_config(args, default_cfg={}, model=None, use_test_size=False, verbose=False):
    new_config = {}
    default_cfg = default_cfg
    # if not default_cfg and model is not None and hasattr(model, 'default_cfg'):
    #     default_cfg = model.default_cfg

    # Resolve input/image size
    in_chans = 3
    if 'chans' in args and args['chans'] is not None:
        in_chans = args['chans']

    input_size = (in_chans, 224, 224)
    if 'input_size' in args and args['input_size'] is not None:
        assert isinstance(args['input_size'], (tuple, list))
        assert len(args['input_size']) == 3
        input_size = tuple(args['input_size'])
        in_chans = input_size[0]  # input_size overrides in_chans
    elif 'img_size' in args and args['img_size'] is not None:
        assert isinstance(args['img_size'], int)
        input_size = (in_chans, args['img_size'], args['img_size'])
    else:
        if use_test_size and 'test_input_size' in default_cfg:
            input_size = default_cfg['test_input_size']
        elif 'input_size' in default_cfg:
            input_size = default_cfg['input_size']
    new_config['input_size'] = input_size

    # resolve interpolation method
    new_config['interpolation'] = 'bicubic'
    if 'interpolation' in args and args['interpolation']:
        new_config['interpolation'] = args['interpolation']
    elif 'interpolation' in default_cfg:
        new_config['interpolation'] = default_cfg['interpolation']

    # resolve dataset + model mean for normalization
    new_config['mean'] = IMAGENET_DEFAULT_MEAN
    if 'mean' in args and args['mean'] is not None:
        mean = tuple(args['mean'])
        if len(mean) == 1:
            mean = tuple(list(mean) * in_chans)
        else:
            assert len(mean) == in_chans
        new_config['mean'] = mean
    elif 'mean' in default_cfg:
        new_config['mean'] = default_cfg['mean']

    # resolve dataset + model std deviation for normalization
    new_config['std'] = IMAGENET_DEFAULT_STD
    if 'std' in args and args['std'] is not None:
        std = tuple(args['std'])
        if len(std) == 1:
            std = tuple(list(std) * in_chans)
        else:
            assert len(std) == in_chans
        new_config['std'] = std
    elif 'std' in default_cfg:
        new_config['std'] = default_cfg['std']

    # resolve default crop percentage
    crop_pct = DEFAULT_CROP_PCT
    if 'crop_pct' in args and args['crop_pct'] is not None:
        crop_pct = args['crop_pct']
    else:
        if use_test_size and 'test_crop_pct' in default_cfg:
            crop_pct = default_cfg['test_crop_pct']
        elif 'crop_pct' in default_cfg:
            crop_pct = default_cfg['crop_pct']
    new_config['crop_pct'] = crop_pct

    if verbose:
        _logger.info('Data processing configuration for current model + dataset:')
        for n, v in new_config.items():
            _logger.info('\t%s: %s' % (n, str(v)))

    return new_config
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in args.schedule:
        args.lr = args.lr * args.decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr