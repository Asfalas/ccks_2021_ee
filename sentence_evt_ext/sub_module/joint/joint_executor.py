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
from sub_module.common.common_seq_tag_executor import *

class JointModelExecutor(CommonSeqTagExecutor):
    def __init__(self, model, train_dataset, dev_dataset, test_dataset, conf):
        super(JointModelExecutor, self).__init__(model, train_dataset, dev_dataset, test_dataset, conf)
        label1 = conf.get('schema', {}).get('label1', [])
        self.label1 = ['O']
        for i in label1:
            self.label1 += ['B-' + i, 'I-' + i]

        label2 = conf.get('schema', {}).get('label2', [])
        self.label2 = ['O']
        for i in label2:
            self.label2 += ['B-' + i, 'I-' + i]


    def get_loss(self, pred, label):
        label1 = label[0].view(-1)
        label2 = label[1].view(-1)
        pred1 = pred[0].view(label1.size()[0], -1)
        pred2 = pred[1].view(label2.size()[0], -1)
        loss = 0.5 * self.loss_function(pred1, label1)
        loss += 0.5 * self.loss_function(pred2, label2)
        return loss
    
    def optimize(self):
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_metric(self, y_pred, y_true, average='micro'):
        labels1 = [i for i in range(1, len(self.label1))]
        labels2 = [i for i in range(1, len(self.label2))]
        precision1, recall1, f11 = calc_metrics(y_true[0], y_pred[0], labels1, average=average)
        precision2, recall2, f12 = calc_metrics(y_true[1], y_pred[1], labels2, average=average)
        precision = (precision1 + precision2) / 2
        recall = (recall1 + recall2) / 2
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        metric = f1
        return precision, recall, f1, metric

    def get_result(self, pred, label, input_data, ignore_index=-100):
        label1 = label[0].view(-1)
        label2 = label[1].view(-1)
        if not self.use_crf:
            pred1 = pred[0].view(label1.size()[0], -1)
            pred2 = pred[1].view(label2.size()[0], -1)
            proba1 = torch.log_softmax(pred1, dim=1)
            proba2 = torch.log_softmax(pred2, dim=1)
            pred_cls1 = torch.argmax(proba1, dim=1)
            pred_cls2 = torch.argmax(proba2, dim=1)
            pred_cls1 = pred_cls1.cpu().view(-1).numpy()
            pred_cls2 = pred_cls2.cpu().view(-1).numpy()
        else:
            pred_cls = self.model.decode(pred)
            pred_cls1 = [i for l in pred_cls[0] for i in l]
            pred_cls2 = [i for l in pred_cls[1] for i in l]

        label1 = label1.cpu().view(-1).numpy()
        label2 = label2.cpu().view(-1).numpy()
        # event_types = input_data[1].cpu().numpy()

        preds = [[], []]
        labels = [[], []]
        for idx, (pred_cls, label) in enumerate(zip([pred_cls1, pred_cls2], [label1, label2])):
            for p, l in zip(pred_cls, label):
                if l == ignore_index:
                    continue
                preds[idx].append(p)
                labels[idx].append(l)
        return preds, labels

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
                label = input_data[-2:]
                # print(input_data)
                if self.use_gpu:
                    inputs = tuple(x.cuda() for x in inputs)
                    label = tuple(x.cuda() for x in label)

                if not self.use_crf:
                    pred = self.model(inputs)
                    # loss calculation
                    loss = self.get_loss(pred, label)
                else:
                    pred, loss = self.model(inputs, label)

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

        logging.info('  训练结束!\n')
        return True

    def eval(self, model, data_loader, mode='dev'):
        with torch.no_grad():
            model.eval()
            bar = tqdm(list(enumerate(data_loader)))
            ground_truth = [[], []]
            pred_label = [[], []]
            for step, input_data in bar:
                # inputs = tuple(x.to(self.device) for x in input_data[:-1])
                inputs = input_data[:-2]
                label = input_data[-2:]
                if self.use_gpu:
                    inputs = tuple(x.cuda() for x in inputs)
                    label = tuple(x.cuda() for x in label)

                if not self.use_crf:
                    pred = self.model(inputs)
                else:
                    pred, loss = self.model(inputs, label)

                y_pred, y_true = self.get_result(pred, label, input_data)
                ground_truth[0] += y_true[0]
                pred_label[0] += y_pred[0]
                ground_truth[1] += y_true[1]
                pred_label[1] += y_pred[1]
            precision, recall, f1, metric = self.get_metric(pred_label, ground_truth)
            logging.info(f"{mode}:{self.epoch}: pre:{round(precision, 3)} rec:{round(recall, 3)} f1:{round(f1, 3)}")
        return metric, None

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
        test_contents = [json.loads(line) for line in open(self.test_path)]
        test_evts = json.load(open("output/evt_men_test.json"))
        offset = 0
        # test
        with torch.no_grad():
            self.model.eval()
            bar = tqdm(list(enumerate(data_loader)))
            for step, input_data in bar:
                # inputs = tuple(x.to(self.device) for x in input_data[:-1])
                inputs = input_data[:-2]
                label = input_data[-2:]
                if self.use_gpu:
                    inputs = tuple(x.cuda() for x in inputs)
                    label = tuple(x.cuda() for x in label)

                if not self.use_crf:
                    pred = self.model(inputs)
                else:
                    pred, loss = self.model(inputs, label)

                proba1 = torch.softmax(pred[0], dim=-1)
                pred_cls1 = torch.argmax(proba1, dim=-1).cpu().numpy()

                proba2 = torch.softmax(pred[1], dim=-1)
                pred_cls2 = torch.argmax(proba2, dim=-1).cpu().numpy()
                
                proba1 = proba1.cpu().numpy()
                proba2 = proba2.cpu().numpy()
                
                mention1, mention2 = [], []
                for i in range(pred_cls1.shape[0]):
                    test_info = test_contents[offset+i]
                    test_evt_info = test_evts[offset+i]
                    
                    labels = []
                    for j in range(1, self.max_seq_len):
                        labels.append([self.label1[pred_cls1[i][j]], proba1[i][j][pred_cls1[i][j]]])
                    mention1 = self.get_mention(test_info['text'], labels)

                    for j in range(1, self.max_seq_len):
                        labels.append([self.label2[pred_cls2[i][j]], proba2[i][j][pred_cls2[i][j]]])
                    mention2 = self.get_mention(test_info['text'], labels)
                    
                    test_info['event_list'] = merge_mention(mention1, mention2, test_evt_info)
                
                offset += pred_cls1.shape[0]
        logging.info("存储test结果：" + self.test_output_path)
        json.dump(test_contents, open(self.test_output_path, 'w'), indent=2, ensure_ascii=False)
        return True

    def get_mention(self, text, label):
        label = label[:len(text)]
        idx = 0
        mention = []
        while idx < len(label):
            if label[idx][0][0] == 'B':
                arg_type = label[idx][0][2:]
                beg = idx
                idx += 1
                while idx < len(label) and label[idx][0] == 'I-' + arg_type:
                    idx += 1
                end = idx
                
                score = 0
                for i in range(beg, end):
                    score += label[i][1]
                score /= end-beg
                score = round(score, 4)
                mention.append(str(beg) + '$%$' + text[beg: end] + '$%$' + arg_type + '$%$' + str(score))
            else:
                idx += 1
        return mention
    
    def merge_mention(mention1, mention2, evt_info):
        candidate_evt_types = [x.split("@#@")[0] for x in evt_info.get("mention", [])]
        offsets = []
        evt_type_map = {}
        for x in mention1:
            beg, mention, arg_type, score = *x.split('$%$')
            beg = int(beg)
            end = beg + len(mention)
            score = float(score)
            new_offsets = []
            
            is_valid = True
            for o in offsets:
                if end <= o[1] or beg >= o[2]:
                    new_offsets.append(o)
                else:
                    if score < o[3]:
                        new_offsets.append(o)
                        is_valid = False
            if is_valid:
                new_offsets.append([beg, end, score, arg_type, mention])
                
            offsets = new_offsets
        time_args = []
        for o in offsets:
            beg, end, score, arg_type, mention = *o
            if arg_type == '时间':
                time_args.append(o)
                continue
            evt_type, role = *arg_type.split('@#@')
            if evt_type not in evt_type_map:
                evt_type_map[evt_type] = []
            evt_type_map[evt_type].append({
                'argument': mention,
                'argument_role': role,
                'argument_start_index': int(beg)
            })
        if evt_type_map:
            for ta in time_args:
                beg, end, score, arg_type, mention = *ta
                for e in evt_type_map:
                    evt_type_map[e].append({
                        'argument': mention,
                        'argument_role': '时间',
                        'argument_start_index': int(beg)
                    })
        else:
            for e in candidate_evt_types:
                for ta in time_args:
                    beg, end, score, arg_type, mention = *ta
                    if e not in evt_type_map:
                        evt_type_map[e] = []
                    evt_type_map[e].append({
                        'argument': mention,
                        'argument_role': '时间',
                        'argument_start_index': int(beg)
                    })
        event_list = []
        for k, v in evt_type_map.items():
            event_list.append({
                "event_type": k,
                "arguments": v
            })
        return event_list
            
                    
                    
            
        
        