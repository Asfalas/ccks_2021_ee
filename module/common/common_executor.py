import sys
sys.path.append('./')
import time
import torch
import json
import logging

from tqdm import tqdm
from common.const import *
from common.utils import *
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.optim import lr_scheduler


class CommonExecutor(object):
    def __init__(self, model, train_dataset, dev_dataset, test_dataset, conf):
        self.use_gpu = conf.get('use_gpu', 0)
        if self.use_gpu:
            model = nn.DataParallel(model)
            model = model.cuda()
        self.model = model
        self.lr = conf.get('lr', 1e-5)
        self.epochs = conf.get('epochs', 20)
        self.model_save_path = conf.get('model_save_path', 'cache/ace05/{metric}_{epoch}.tar')
        self.best_model_save_path = conf.get('best_model_save_path', 'cache/ace05/best.json')
        self.best_model_config_path = conf.get('best_model_config_path', 'cache/ace05/best_config.json')
        self.max_seq_len = conf.get('max_seq_len', 400)
        self.test_path = conf.get('test_path', 'data/duee_test1.json')
        self.label = conf.get('label', ['O', 'B', 'I'])
        self.use_crf = conf.get('use_crf', 0)
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dev_dataset = dev_dataset
        self.test_output_path = conf.get('test_output_path', 'output/arg_men_test.json')
        self.weight_decay = conf.get('weight_decay', 0)
        self.batch_size = conf.get('batch_size', 32)
        self.accumulate_step = conf.get('accumulate_step', 1)
        self.lr_gamma = conf.get('lr_gamma', 0.9)
        self.early_stop_threshold = conf.get('early_stop_threshold', 5)
        self.best_metric = 0
        self.optimizer = None
        self.scheduler = None
        self.loss_function = None
        self.epoch = 0
        self.early_stop = 0
        self.conf = conf
        setup_seed(seed)

    def setup(self):
        total_params = self.model.parameters() if not self.use_gpu else self.model.module.parameters()
        self.optimizer = torch.optim.Adam(total_params, lr=self.lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_gamma)
        self.loss_function = nn.CrossEntropyLoss()
    
    def load_model(self, checkpoint):
        if 'json' in checkpoint:
            real_model = json.load(open(checkpoint))['best_model']
        else:
            real_model = checkpoint
        logging.info('  加载模型参数: ' + str(real_model))
        checkpoint_info = torch.load(real_model, map_location=torch.device('cpu'))
        if self.use_gpu:
            self.model.module.load_state_dict(checkpoint_info['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint_info['model_state_dict'])
    
    def optimize(self):
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save_checkpoint(self, model, metric):
        path = self.model_save_path.format(metric=round(metric, 4), epoch=self.epoch)
        model_state_dict = model.state_dict() if not self.use_gpu else model.module.state_dict()
        logging.info('  存储模型参数: ' + str(path))
        result = {
            'model_state_dict': model_state_dict,
        }
        torch.save(result, path)
        json.dump({'best_model': path}, open(self.best_model_save_path, 'w', encoding='utf-8'))
        json.dump(self.conf, open(self.best_model_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    def train(self, checkpoint=None):
        raise NotImplementedError()

    def eval(self, model, data_loader, mode='dev'):
        raise NotImplementedError()

    def test(self, dataset=None, checkpoint=None):
        raise NotImplementedError()
