import sys
sys.path.append('./')
sys.path.append('../sentence_evt_ext/')
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
from collections import Counter

class ArtMultiTaggerModelExecutor(CommonSeqTagExecutor):
    def __init__(self, model, train_dataset, dev_dataset, test_dataset, conf):
        super(ArtMultiTaggerModelExecutor, self).__init__(model, train_dataset, dev_dataset, test_dataset, conf)
        self.max_seq_len = conf.get('max_seq_len', 256)
        self.real_max_seq_len = self.max_seq_len - 4 
        self.pretrained_model_name = conf.get('pretrained_model_name', 'bert-base-chinese')
        # self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
        self.max_ent_len = conf.get("max_ent_len", 10)
        self.max_evt_len = conf.get("max_evt_len", 3)
        self.role_list = conf.get("role_list", [])
        self.label_map = conf.get("label_map", {})
        self.event_list = list(self.label_map.keys())
        self.use_crf = conf.get('use_crf', 0)
        self.use_lstm = conf.get('use_lstm', 0)
        self.retrain = conf.get("retrain", 0)
        self.enum_list = conf.get("enum_list", [])
        self.is_multi_tagger = conf.get("is_multi_tagger", True)

    def get_loss(self, logits, labels):
        count = 0
        loss = 0
        for i in range(0, len(self.event_list)):
            logit = logits[i]
            label = labels[i]
            if logit.numel():
                loss += self.loss_function(logit, label)
                count += 1
        loss /= count
        return loss
    
    def optimize(self):
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_metric(self, y_pred, y_true, average='micro'):
        
        precision = []
        recall = []
        f1 = []
        
        allow_labels = []
        for e in self.event_list:
            allow_labels.append([x for x in range(1, 2 * len(self.label_map[e]) + 1)])

        for pred, true, allow_label in zip(y_pred, y_true, allow_labels):
#             is_all_zero = True
#             for i in true:
#                 if i!=0:
#                     is_all_zero = False
#                     break
#             if is_all_zero:
#                 continue
            if pred and true:
                p, r, f = calc_metrics(true, pred, allow_label, average=average)
                precision.append(p)
                recall.append(r)
                f1.append(f)

#         pre = sum(precision) / len(precision)
#         re = sum(recall) / len(recall)
        f1_val = sum(f1) / len(f1)

        metric = f1_val
        return precision, recall, f1, metric

    def get_result(self, logits, labels, ignore_index=-100):
        final_preds = tuple()
        final_labels = tuple()

        for pred, label in zip(logits, labels):
            if pred.numel():
                label = label.view(-1)
                pred = pred.view(label.size()[0], -1)
                proba = torch.log_softmax(pred, dim=1)
                pred_cls = torch.argmax(proba, dim=1)
                pred_cls = pred_cls.cpu().view(-1).numpy()
                label = label.cpu().numpy()
            else:
                pred_cls = []
                label = []
            tmp_preds, tmp_labels = [], []
            for idx, (p, l) in enumerate(zip(pred_cls, label)):
                if l == ignore_index:
                    continue
                tmp_preds.append(p)
                tmp_labels.append(l)
            final_preds += (tmp_preds, )
            final_labels += (tmp_labels, )
            
        return final_preds, final_labels

    def train(self, checkpoint=None):
        logging.info('  开始训练')
        train_data_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=0)
        dev_data_loader = torch.utils.data.DataLoader(dataset=self.dev_dataset, batch_size=self.batch_size,
                                                      shuffle=True, num_workers=0)
        # init
        if self.use_lstm or self.is_multi_tagger:
            logging.info('  锁定bert参数')
            bert_params = self.model.module.bert.parameters() if self.use_gpu else self.model.bert.parameters()
            for i in bert_params:
                i.requires_grad = False
        if self.retrain:
            checkpoint = self.best_model_save_path
            self.load_model(checkpoint)
            logging.info('  释放bert参数')
            bert_params = self.model.module.bert.parameters() if self.use_gpu else self.model.bert.parameters()
            for i in bert_params:
                i.requires_grad = True
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
                # print(input_data)
                if self.use_gpu:
                    inputs = tuple(x.cuda() for x in inputs)
                    labels = tuple(x.cuda() for x in labels)

                logits = self.model(inputs)

                # 取mask
                sample_flags = labels[0]
                labels = labels[1]
                label_vector = []
                logit_vector = []
                for i in range(len(self.event_list)):
                    logit = logits[i]
                    label = labels[:, i, :]
                    logit_mask = sample_flags[:, i].unsqueeze(dim=-1).unsqueeze(dim=-1)
                    label_mask = sample_flags[:, i].unsqueeze(dim=-1)
                    logit = torch.masked_select(logit, logit_mask)
                    label = torch.masked_select(label, label_mask)
                    if logit.numel() or label.numel():
                        label = label.view(-1)
                        logit = logit.view(label.size()[0], -1)
                    label_vector.append(label)
                    logit_vector.append(logit)

                loss = self.get_loss(logit_vector, label_vector)
                loss.backward()
                
                # params optimization
                if ((step + 1) % self.accumulate_step) == self.accumulate_step - 1 or step == len(
                        train_data_loader) - 1:
                    self.optimize()

                # calc metric info
                y_pred, y_true = self.get_result(logit_vector, label_vector)
                precision, recall, f1, metric = self.get_metric(y_pred, y_true)
                bar.set_description(f"step:{step}: f1:{round(metric, 3)} loss:{loss}")

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
            ground_truth = [[] for i in range(len(self.event_list)+1)]
            pred_label = [[] for i in range(len(self.event_list)+1)]
            for step, input_data in bar:
                inputs = input_data[:-2]
                labels = input_data[-2:]
                # print(input_data)
                if self.use_gpu:
                    inputs = tuple(x.cuda() for x in inputs)
                    labels = tuple(x.cuda() for x in labels)

                logits = self.model(inputs)

                labels = labels[1]
                label_vector = []
                logit_vector = []
                for i in range(len(self.event_list)):
                    logit = logits[i]
                    label = labels[:, i, :]

                    label = label.reshape(-1)
                    logit = logit.reshape(label.size()[0], -1)
                    label_vector.append(label)
                    logit_vector.append(logit)

                y_pred, y_true = self.get_result(logit_vector, label_vector)

                for i in range(len(y_pred)):
                    ground_truth[i] += y_true[i]
                    pred_label[i] += y_pred[i]

            precision, recall, f1, metric = self.get_metric(pred_label, ground_truth)
            for i, name in enumerate(self.event_list):
                logging.info(f"{name}:{self.epoch}: pre:{round(precision[i], 3)} rec:{round(recall[i], 3)} f1:{round(f1[i], 3)}")
            logging.info(f"{mode}:{self.epoch}: {round(metric, 3)}")
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
        
        offset = 0
        # test
        with torch.no_grad():
            self.model.eval()
            bar = tqdm(list(enumerate(data_loader)))
            for step, input_data in bar:
                # inputs = tuple(x.to(self.device) for x in input_data[:-1])
                inputs = input_data
                if self.use_gpu:
                    inputs = tuple(x.cuda() for x in inputs)
                    result = self.model.module.predict(inputs)
                else:
                    result = self.model.predict(inputs, use_gpu=False)

                for i in range(len(result)):
                    test_info = test_contents[offset+i]
                    event_list = self.handle_result(result[i], test_info)
                    if event_list:
                        test_info['event_list'] = event_list
                
                offset += len(result)
        logging.info("存储test结果：" + self.test_output_path)
        import jsonlines
        with jsonlines.open(self.test_output_path, mode='w') as writer:
            for i in test_contents:
                if 'text' in i:
                    del i['text']
                writer.write(i)
#         json.dump(test_contents, open(self.test_output_path, 'w'), indent=2, ensure_ascii=False)
        return True
    
    def handle_result(self, result, test_info):
        text = test_info['text']
        event_list = []
        if not result:
            return []
        evt_arg_map = {}
        for k in result:
            beg, end, evt_id, role_id = k
            evt_type = self.event_list[evt_id]
            role = self.label_map[evt_type][role_id]
            argument = text[beg: end]
            if not argument:
                continue
            if evt_type not in evt_arg_map:
                evt_arg_map[evt_type] = set()
            evt_arg_map[evt_type].add('@#@'.join([argument, role]))
        
        for et, arg_set in evt_arg_map.items():
            arguments = []
            for arg_info in arg_set:
                argument, role = arg_info.split('@#@')[0], arg_info.split('@#@')[1]
                arguments.append({
                    "argument": argument,
                    "role": role
                })
            event = {
                "event_type": et,
                "arguments": arguments
            }
            event_list.append(event)

        return event_list
