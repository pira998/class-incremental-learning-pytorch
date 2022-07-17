from collections import defaultdict, deque
import os
import warnings
from xml.dom import NotSupportedErr
import numpy as np

import torch
from torchvision import transforms

from imagenet import ImageNet1000
from resnet import resnet20, resnet32, resnet44, resnet56

from continuum import ClassIncremental
from continuum.datasets import CIFAR100, ImageFolderDataset

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

try:
    interpolation = torch.transforms.functional.InterpolationMode.BICUBIC
except:
    interpolation = 3


def get_backbone(args):
    '32, 20, 44, 56'
    if args.backbone == 'resnet32':
        return resnet32()
    elif args.backbone == 'resnet20':
        return resnet20()
    elif args.backbone == 'resnet44':
        return resnet44()
    elif args.backbone == 'resnet56':
        return resnet56()

    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

    else:
        print('Not using distributed mode')
        args.distributed = False
        raise NotSupportedErr('Not using distributed mode')
        return
    
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                            world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    print('| distributed init completed', flush=True)
    setup_for_distributed(args.rank == 0)

def is_main_process():
    return torch.distributed.get_rank() == 0

def setup_for_distributed(is_master):
    """
    This function disables printing if not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
    
def init_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_dataset(is_train,args):
    transform = build_transform(is_train, args)
    if args.data_set.lower() == 'cifar':
        dataset = CIFAR100(args.data_path, train=is_train, download=True)
    elif args.data_set.lower() == 'imagenet1000':
        dataset = ImageNet1000(args.data_path, train=is_train)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.data_set))
    

    scenario = ClassIncremental(
        dataset,
        initial_increment=args.num_bases,
        increment = args.increment,
        transformations = transform.transforms,
        class_order = args.class_order,
    )
    nb_classes = scenario.nb_classes
    return scenario, nb_classes

def build_transform(is_train, args):
    if args.aa == 'none':
        args.aa = None
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resize_im = args.input_size > 32 # if input_size is larger than 32, we need to resize
        if is_train:
            transform = create_transform(
                input_size = args.input_size,
                is_training = True,
                color_jitter = args.color_jitter, 
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob = args.reprob,
                re_mode = args.remode,
                re_count = args.recount,
            )
            if not resize_im:
                transform.transforms[0] = transforms.RandomCrop(
                    args.input_size,
                    padding=4
                )
            if args.input_size == 32 and args.data_set == 'CIFAR':
                transform.transforms[-1] = transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            
            return transform
        
        t = []

        if resize_im:
            size = int((256/224)*args.input_size)
            t.append(transforms.Resize(size, interpolation=interpolation))
            t.append(transforms.CenterCrop(args.input_size))
        
        t.append(transforms.ToTensor())
        if args.data_set == 'CIFAR' and args.input_size ==32:
            t.append(transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ))
        else:
            t.append(transforms.Normalize(
                IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            ))
        
        return transforms.Compose(t)

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a window or the global series average."""

    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return np.max(self.deque)

    @property
    def min(self):
        return np.min(self.deque)

    @property
    def std(self):
        return np.std(self.deque)

    @property
    def count(self):
        return len(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    @property
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            min=self.min,
            count=self.count,
            value=self.value,
            std=self.std,
        )

class MetricLogger(object):   # encapsulate the logic of logging
    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(SmoothedValue) # a dict of SmoothedValue objects
        self.delimiter = delimiter 

    def update(self, **kwargs):  # update the meters with the values passed in
        for k, v in kwargs.items(): # for each key, value pair
            if v is None:
                continue
            if isinstance(v, torch.Tensor): #
                v = v.item() # convert tensor to scalar
            assert isinstance(v, (float, int)) # make sure it's a number
            self.meters[k].update(v) # update the meter with the new value

    def update_dict(self, d):
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
        
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append('{}: {:.4f}'.format(name, meter.val))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter
    