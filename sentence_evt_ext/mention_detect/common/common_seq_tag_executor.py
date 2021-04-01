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


class CommonSeqTagExecutor(object):
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

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dev_dataset = dev_dataset
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

    def get_loss(self, pred, label):
        label = label.view(-1)
        pred = pred.view(label.size()[0], -1)
        loss = self.loss_function(pred, label)
        return loss
    
    def optimize(self):
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_metric(self, y_pred, y_true, average='micro'):
        precision, recall, f1 = calc_metrics(y_true, y_pred, average=average)
        metric = f1
        return precision, recall, f1, metric

    def get_result(self, pred, label, input_data, ignore_index=-100):
        label = label.view(-1)
        pred = pred.view(label.size()[0], -1)
        proba = torch.log_softmax(pred, dim=1)
        pred_cls = torch.argmax(proba, dim=1)

        label = label.cpu().view(-1).numpy()
        pred_cls = pred_cls.cpu().view(-1).numpy()
        # event_types = input_data[1].cpu().numpy()

        preds, labels = [], []
        for idx, (p, l) in enumerate(zip(pred_cls, label)):
            if l == ignore_index:
                continue
            preds.append(p)
            labels.append(l)
        return preds, labels

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
        logging.info('  开始训练')
        train_data_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=0)
        dev_data_loader = torch.utils.data.DataLoader(dataset=self.dev_dataset, batch_size=self.batch_size,
                                                      shuffle=True, num_workers=0)
        # init
        self.setup()
        # # load checkpoint when needed
        # if checkpoint != '' and checkpoint is not None:
        #     self.load_checkpoint(checkpoint)

        # train
        while self.epoch < self.epochs:
            logging.info('  epoch: ' + str(self.epoch))

            self.model.train()
            # output lr info
            for lrs in self.optimizer.state_dict()['param_groups']:
                logging.info('  学习率: ' + str(lrs['lr']))
            # training loop
            bar = tqdm(list(enumerate(train_data_loader)))
            for step, input_data in bar:
                inputs = input_data[:-1]
                label = input_data[-1]
                if self.use_gpu:
                    inputs = tuple(x.cuda() for x in inputs)
                    label = label.cuda()

                pred = self.model(inputs)
                if isinstance(pred, tuple):
                    pred = pred[0]

                # loss calculation
                loss = self.get_loss(pred, label)

                loss.backward()
                
                # params optimization
                if ((step + 1) % self.accumulate_step) == self.accumulate_step - 1 or step == len(
                        train_data_loader) - 1:
                    self.optimize()

                # calc metric info
                y_pred, y_true = self.get_result(pred, label, input_data)
                precision, recall, f1, metric = self.get_metric(y_pred, y_true)
                bar.set_description(f"step:{step}: pre:{round(precision, 3)} loss:{loss}")

            # adjust learning rate
            self.scheduler.step()
            # evaluate on dev dataset
            metric, _ = self.eval(self.model, dev_data_loader)
            # model save when achieve best metric
            if metric > self.best_metric:
                self.save_checkpoint(self.model, metric)
                self.best_metric = metric
                self.early_stop = 0
            else:
                self.early_stop += 1
            # early stop
            if self.early_stop > self.early_stop_threshold:
                break
            self.epoch += 1
            self.post_epoch(self.epoch)

        logging.info('  训练结束!\n')
        return True

    def eval(self, model, data_loader, mode='dev'):
        with torch.no_grad():
            model.eval()
            bar = tqdm(list(enumerate(data_loader)))
            ground_truth = []
            pred_label = []
            pred_results = []
            for step, input_data in bar:
                # inputs = tuple(x.to(self.device) for x in input_data[:-1])
                inputs = input_data[:-1]
                label = input_data[-1]
                if self.use_gpu:
                    inputs = tuple(x.cuda() for x in inputs)

                pred = model(inputs)
                if isinstance(pred, tuple):
                    pred = pred[0]
                y_pred, y_true = self.get_result(pred, label, input_data)
                ground_truth += y_true
                pred_label += y_pred
                pred_results.append(y_pred)
            precision, recall, f1, metric = self.get_metric(pred_label, ground_truth)
            logging.info(f"{mode}:{self.epoch}: pre:{round(precision, 3)} rec:{round(recall, 3)} f1:{round(f1, 3)}")
        return metric, pred_results

    # def test(self, dataset=None, checkpoint=None):
    #     logging.info('  测试开始')
    #     if not dataset:
    #         dataset = self.test_dataset
    #     # load checkpoint
    #     if not checkpoint:
    #         checkpoint = self.best_model_save_path
    #     self.load_model(checkpoint)
    #     # load test dataset
    #     data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size,
    #                                                    shuffle=False, num_workers=0)
    #     # evaluate on test dataset
    #     metric, pred_results = self.eval(self.model, data_loader, mode='test')
    #     logging.info("  测试结果  f1: " + str(metric))
    #     logging.info("  测试结束!\n")
    #     return True
