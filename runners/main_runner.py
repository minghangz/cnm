import collections
import logging
import os
import time

import numpy as np
import torch
from torch.utils import data

from models.loss import ivc_loss, rec_loss
from utils import TimeMeter, AverageMeter

import pickle
import copy
from pathlib import Path


def info(msg):
    print(msg)
    logging.info(msg)


class MainRunner:
    def __init__(self, args):
        self.args = args
        self._build_dataset()

        self.args['model']['config']['max_num_words'] = self.args['dataset']['max_num_words']
        self.args['model']['config']['frames_input_size'] = self.args['dataset']['frame_dim']
        self.args['model']['config']['words_input_size'] = self.args['dataset']['word_dim']
        self.args['model']['config']['vocab_size'] = self.train_set.vocab_size

        self._build_model()
        if 'train' in args:
            self._build_optimizer()
            self.num_updates = 0

    def train(self):
        best_results = None
        for epoch in range(self.args['train']['max_num_epochs']):
            info('Start Epoch {}'.format(epoch))
            self.model_saved_path = self.args['train']['model_saved_path']
            os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
            save_path = os.path.join(self.model_saved_path, 'model-{}.pt'.format(epoch))

            self._train_one_epoch(epoch)
            self._save_model(save_path)
            results = self.eval()
            if best_results is None or results['mIoU'].avg > best_results['mIoU'].avg:
                best_results = results
                os.system('cp %s %s'%(save_path, os.path.join(self.model_saved_path, 'model-best.pt')))
                info('Best results have been updated.')
            info('=' * 60)
        
        msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in best_results.items()])
        info('Best results:')
        info('|'+msg+'|')


    def _train_one_epoch(self, epoch, **kwargs):
        self.model.train()

        def print_log():
            msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, bid, curr_lr)
            for k, v in loss_meter.items():
                msg += '{} = {:.4f}, '.format(k, v.avg)
                v.reset()
            msg += '{:.3f} seconds/batch'.format(1.0 / time_meter.avg)
            info(msg)

        display_n_batches, bid = 50, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())

        for bid, batch in enumerate(self.train_loader, 1):
            self.model.froze_mask_generator()
            self.rec_optimizer.zero_grad()
            net_input = move_to_cuda(batch['net_input'])
            output = self.model(**net_input)
            loss, loss_dict = rec_loss(**output, **self.args['loss'])
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.rec_optimizer.step()

            self.model.froze_reconstructor()
            self.mask_optimizer.zero_grad()
            output = self.model(**net_input)
            loss, ivc_loss_dict = ivc_loss(**output, **self.args['loss'])
            loss_dict.update(ivc_loss_dict)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.mask_optimizer.step()

            self.num_updates += 1
            curr_lr = self.rec_lr_scheduler.step_update(self.num_updates)
            self.mask_lr_scheduler.step_update(self.num_updates)
            time_meter.update()
            for k, v in loss_dict.items():
                loss_meter[k].update(v)

            if bid % display_n_batches == 0:
                print_log()

        if bid % display_n_batches != 0:
            print_log()

    def eval(self, data_loader=None):
        self.model.eval()
        if data_loader is None:
            data_loader = self.test_loader
        with torch.no_grad():
            metrics_logger = collections.defaultdict(lambda: AverageMeter())

            with torch.no_grad():
                for bid, batch in enumerate(data_loader, 1):
                    durations = np.asarray([i[1] for i in batch['raw']])
                    gt = np.asarray([i[2] for i in batch['raw']])
                    bsz = len(durations)

                    net_input = move_to_cuda(batch['net_input'])
                    time_st = time.time()
                    output = self.model(**net_input)
                    width = output['width'].view(bsz)
                    center = output['center'].view(bsz)
                    selected_props = torch.stack([torch.clamp(center-width/2, min=0), 
                                                  torch.clamp(center+width/2, max=1)], dim=-1)
                    time_en = time.time()
                    selected_props = selected_props.cpu().numpy()
                    
                    gt = gt / durations[:, np.newaxis]
                    res = top_1_metric(selected_props, gt)
                    for key, v in res.items():
                        metrics_logger[key].update(v, bsz)
                    metrics_logger['Time'].update(time_en-time_st, bsz)

            msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in metrics_logger.items()])
            info('|'+msg+'|')

            return metrics_logger

    def _build_dataset(self):
        import datasets as da
        import pickle
        from torch.utils.data import DataLoader
        args = self.args['dataset']
        cls = getattr(da, args['dataset'], None)

        with open(args['vocab_path'], 'rb') as fp:
            vocab = pickle.load(fp)
        self.train_set = cls(data_path=args['train_data'], vocab=vocab, args=args, is_training=True)
        self.test_set = cls(data_path=args['test_data'], vocab=vocab, args=args)
        self.val_set = cls(data_path=args['val_data'], vocab=vocab, args=args) if args['val_data'] else None
        info('train: {} samples, test: {} samples'.format(len(self.train_set), len(self.test_set)))
        batch_size = self.args['train']['batch_size']

        def worker_init_fn(worker_id):
            def set_seed(seed):
                import random
                import numpy as np
                import torch

                random.seed(seed)
                np.random.seed(seed + 1)
                torch.manual_seed(seed + 3)
                torch.cuda.manual_seed(seed + 4)
                torch.cuda.manual_seed_all(seed + 4)

            set_seed(8 + worker_id)

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                       collate_fn=self.train_set.collate_data, num_workers=2,
                                       worker_init_fn=worker_init_fn)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False,
                                      collate_fn=self.test_set.collate_data,
                                      num_workers=1)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False,
                                     collate_fn=self.val_set.collate_data,
                                     num_workers=1) if args['val_data'] else None

    def _build_model(self):
        model_config = self.args['model']
        import models

        self.model = getattr(models, model_config['name'], None)(model_config['config'])
        self.model = self.model.cuda()
        print(self.model)

    def _build_optimizer(self):
        from optimizers import AdamOptimizer
        from optimizers.lr_schedulers import InverseSquareRootSchedule

        self.model.froze_mask_generator()
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        args = self.args['train']["reconstructor"]
        self.rec_optimizer = AdamOptimizer(args, parameters)
        self.rec_lr_scheduler = InverseSquareRootSchedule(args, self.rec_optimizer)

        self.model.froze_reconstructor()
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        args = self.args['train']["generator"]
        self.mask_optimizer = AdamOptimizer(args, parameters)
        self.mask_lr_scheduler = InverseSquareRootSchedule(args, self.mask_optimizer)
        

    def _save_model(self, path):
        state_dict = {
            'num_updates': self.num_updates,
            'config': self.args,
            'model_parameters': self.model.state_dict(),
        }
        torch.save(state_dict, path)
        info('save model to {}, num_updates {}.'.format(path, self.num_updates))

    def _load_model(self, path):
        state_dict = torch.load(path)
        self.num_updates = state_dict['num_updates']
        self.mask_lr_scheduler.step_update(self.num_updates)
        self.rec_lr_scheduler.step_update(self.num_updates)
        parameters = state_dict['model_parameters']
        self.model.load_state_dict(parameters)
        info('load model from {}, num_updates {}.'.format(path, self.num_updates))


def calculate_IoU_batch2(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    # iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def top_n_metric(preds, label):
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU_batch2((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def top_1_metric(pred, label):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU_batch2((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)
