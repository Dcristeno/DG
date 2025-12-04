'''
Reference:
Official PyTorch implementation of the paper Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval. (CVPR 2024)
Modified for NCR (NeurIPS 2021) noisy correspondence + Weights & Biases logging
'''
import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size, get_rank

from .bases import ImageDataset, TextDataset, ImageTextDataset

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid

import TensorSaver

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
            batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict


def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)
    # Preserve original NCR logging hook (if used)
    if hasattr(dataset.train[0], '__len__') and len(dataset.train[0]) > 2:
        TensorSaver.pidstatistics = dataset.train[0][-1]

    if args.training:
        train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)

        # ✅ NCR: Inject real_correspondences if not already present
        # Support both `noise_ratio` (NCR official) and `noisy_rate` (your code)
        noise_ratio = getattr(args, 'noise_ratio', 0.0)
        if hasattr(args, 'noisy_rate'):
            noise_ratio = args.noisy_rate

        if not hasattr(dataset, 'real_correspondences'):
            logger.info(f"Generating real_correspondences with noise_ratio={noise_ratio:.2f}")
            total = len(dataset.train)
            clean_num = int(total * (1 - noise_ratio))
            # Simulate: first `clean_num` clean, rest noisy → shuffle
            real_corr = torch.cat([
                torch.ones(clean_num, dtype=torch.bool),
                torch.zeros(total - clean_num, dtype=torch.bool)
            ])
            real_corr = real_corr[torch.randperm(total)]  # shuffle indices
            dataset.real_correspondences = real_corr
        else:
            logger.info("real_correspondences already exists in dataset.")

        train_set = ImageTextDataset(dataset.train, args,
                                     train_transforms,
                                     text_length=args.text_length)
        # ✅ Link back to raw dataset so `loader.dataset.raw_dataset.real_correspondences` works
        train_set.raw_dataset = dataset

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)
                train_loader = DataLoader(train_set,
                                          batch_sampler=batch_sampler,
                                          num_workers=num_workers,
                                          collate_fn=collate,
                                          pin_memory=True)
            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, '
                    f'id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate,
                                          pin_memory=True)
        elif args.sampler == 'random':
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate,
                                      pin_memory=True)
        else:
            logger.error('unsupported sampler! expected identity or random but got {}'.format(args.sampler))
            raise ValueError(f"Unknown sampler: {args.sampler}")

        # use test set as validate set
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                   val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                  ds['captions'],
                                  text_length=args.text_length)

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True)

        # ✅ W&B: Log dataset metadata (only master process)
        if (not args.distributed) or (args.distributed and get_rank() == 0):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.config.update({
                        "dataset_name": args.dataset_name,
                        "train_samples": len(dataset.train),
                        "val_samples": len(dataset.val) if hasattr(dataset, 'val') else 0,
                        "test_samples": len(dataset.test),
                        "num_classes": num_classes,
                        "noise_ratio": noise_ratio,
                        "sampler": args.sampler,
                        "img_size": args.img_size,
                        "text_length": args.text_length,
                    }, allow_val_change=True)
                    logger.info("Dataset metadata logged to W&B.")
            except Exception as e:
                logger.warning(f"W&B metadata logging skipped: {e}")

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:
        # build dataloader for testing
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True)
        return test_img_loader, test_txt_loader, num_classes