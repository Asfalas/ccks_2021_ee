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


class ClsModelExecutor(object):
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
        self.max_role_len = conf.get('max_role_len', 7)
        self.max_ent_len = conf.get('max_ent_len', 32)
        self.event_num = conf.get('event_num', 66)
        self.test_path = conf.get('test_path', 'data/duee_test1.json')
        self.label = conf.get('label', ['O', 'B', 'I'])
        self.event_schema = conf.get('event_schema', {})
        self.event_list = conf.get('event_list', [])
        
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

    def get_loss(self, evt_logits, arg_logits, evt_labels, arg_labels):
        new_evt_labels = evt_labels.view(-1)
        new_evt_logits = evt_logits.view(evt_labels.size()[0], -1)
        new_arg_labels = arg_labels.view(-1)
        new_arg_logits = arg_logits.view(new_arg_labels.size()[0], -1)

        loss = self.loss_function(new_evt_logits, new_evt_labels)
        loss += self.loss_function(new_arg_logits, new_arg_labels)
        return loss
    
    def optimize(self):
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_metric(self, y_pred, y_true, average='micro'):
        labels = [i for i in range(1, self.max_role_len * self.event_num)]
        precision, recall, f1 = calc_metrics(y_true, y_pred, labels, average=average)
        metric = f1
        return precision, recall, f1, metric

    def get_result(self, evt_logits, evt_labels, arg_logits, arg_labels, ignore_index=-100):
        evt_labels = evt_labels.view(-1)
        evt_pred = evt_logits.view(evt_labels.size()[0], -1)
        evt_proba = torch.log_softmax(evt_pred, dim=1)
        evt_pred_cls = torch.argmax(evt_proba, dim=1)

        evt_labels = evt_labels.cpu().view(-1).numpy()
        evt_pred_cls = evt_pred_cls.cpu().view(-1).numpy()

        arg_labels = arg_labels.view(-1, 1).squeeze()
        arg_pred = arg_logits.view(arg_labels.size()[0], -1)
        arg_proba = torch.log_softmax(arg_pred, dim=1)
        arg_pred_cls = torch.argmax(arg_proba, dim=1)

        arg_labels = arg_labels.cpu().view(-1).numpy()
        arg_pred_cls = arg_pred_cls.cpu().view(-1).numpy()

        # print(evt_labels.shape)
        # print(evt_pred_cls.shape)
        # print(arg_pred_cls.shape)
        # print(arg_labels.shape)

        preds, labels = [], []
        for idx, (p, l) in enumerate(zip(arg_pred_cls, arg_labels)):
            if l == ignore_index:
                continue
            evt_type_pred = evt_pred_cls[int(idx/self.max_ent_len)]
            evt_type_label = evt_labels[int(idx/self.max_ent_len)]
            if p != 0:
                preds.append(p + evt_type_pred * self.max_role_len)
            else:
                preds.append(0)
            if l != 0: 
                labels.append(l + evt_type_label * self.max_role_len)
            else:
                labels.append(0)
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
                inputs = input_data[:-2]
                labels = input_data[-2:]
                if self.use_gpu:
                    inputs = tuple(x.cuda() for x in inputs)
                    labels = tuple(x.cuda() for x in labels)

                evt_logits, arg_logits = self.model(inputs)
                evt_labels, arg_labels = labels

                # loss calculation
                loss = self.get_loss(evt_logits, arg_logits, evt_labels, arg_labels)

                loss.backward()
                
                # params optimization
                if ((step + 1) % self.accumulate_step) == self.accumulate_step - 1 or step == len(
                        train_data_loader) - 1:
                    self.optimize()

                # calc metric info
                y_pred, y_true = self.get_result(evt_logits, evt_labels, arg_logits, arg_labels)
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
                inputs = input_data[:-2]
                labels = input_data[-2:]
                if self.use_gpu:
                    inputs = tuple(x.cuda() for x in inputs)
                    labels = tuple(x.cuda() for x in labels)

                evt_logits, arg_logits = self.model(inputs)
                evt_labels, arg_labels = labels

                y_pred, y_true = self.get_result(evt_logits, evt_labels, arg_logits, arg_labels)
                ground_truth += y_true
                pred_label += y_pred
                pred_results.append(y_pred)
            precision, recall, f1, metric = self.get_metric(pred_label, ground_truth)
            logging.info(f"{mode}:{self.epoch}: pre:{round(precision, 3)} rec:{round(recall, 3)} f1:{round(f1, 3)}")
        return metric, pred_results

    def test(self, dataset=None, checkpoint=None):
        logging.info('  测试开始')
        if not dataset:
            dataset = self.test_dataset
        # load checkpoint
        if not checkpoint:
            checkpoint = self.best_model_save_path
        self.load_model(checkpoint)
        # load test dataset
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size,
                                                       shuffle=False, num_workers=0)
        test_result = []
        test_contents = json.load(open(self.test_path))
        offset = 0
        # test
        with torch.no_grad():
            self.model.eval()
            bar = tqdm(list(enumerate(data_loader)))
            for step, input_data in bar:
                inputs = input_data[:-2]
                labels = input_data[-2:]
                if self.use_gpu:
                    inputs = tuple(x.cuda() for x in inputs)
                    labels = tuple(x.cuda() for x in labels)

                evt_logits, arg_logits = self.model(inputs)
                evt_labels, arg_labels = labels

                evt_labels = evt_labels.view(-1)
                evt_pred = evt_logits.view(evt_labels.size()[0], -1)
                evt_proba = torch.log_softmax(evt_pred, dim=1)
                evt_pred_cls = torch.argmax(evt_proba, dim=1)

                evt_pred_cls = evt_pred_cls.cpu().view(-1).numpy()

                arg_labels = arg_labels.view(-1, 1).squeeze()
                arg_pred = arg_logits.view(arg_labels.size()[0], -1)
                arg_proba = torch.log_softmax(arg_pred, dim=1)
#                 arg_pred_cls = torch.argmax(arg_proba, dim=1)

                arg_pred_cls = arg_proba.cpu()
                
                for i in range(len(evt_pred_cls)):
                    event_info = test_contents[i+offset]
                    event_type = self.event_list[evt_pred_cls[i]]
                    event_info['event']['event_type'] = event_type
                    for j in range(len(event_info['event']['argument'])):
                        role_list = self.event_schema[event_type]
                        proba = arg_pred_cls[i*self.max_ent_len+j][:len(role_list)]
                        role_index = torch.argmax(proba).cpu().numpy()
                        role = role_list[role_index]
                        event_info['event']['argument'][j]['role'] = role
                offset += len(evt_pred_cls)
        json.dump(test_contents, open(self.test_output_path, 'w'), indent=2, ensure_ascii=False)
        return True

