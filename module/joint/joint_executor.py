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
from module.common.common_executor import *
from collections import Counter

class JointModelExecutor(CommonExecutor):
    def __init__(self, model, train_dataset, dev_dataset, test_dataset, conf):
        super(JointModelExecutor, self).__init__(model, train_dataset, dev_dataset, test_dataset, conf)
        self.max_seq_len = conf.get('max_seq_len', 256)
        self.pretrained_model_name = conf.get('pretrained_model_name', 'bert-base-chinese')
        # self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
        self.max_ent_len = conf.get("max_ent_len", 10)
        self.max_evt_len = conf.get("max_evt_len", 3)
        self.role_list = conf.get("role_list", [])
        self.possible_role_list = conf.get("possible_role_list", [])
        self.eval_output_path = conf.get('eval_output_path', 'output/joint_eval.json')
        self.dev_path = conf.get("dev_path", 'data/tmp_dev.json')
        self.entity_list = conf.get("entity_list", [])
        self.event_list = conf.get("event_list", [])
        self.use_crf = conf.get('use_crf', 0)
        self.use_lstm = conf.get('use_lstm', 0)
        self.retrain = conf.get("retrain", 0)

    def get_loss(self, logits, labels):
        evt_logit, ent_logit, role_logit = logits
        evt_label, ent_label, role_label = labels
        
        evt_label = evt_label.view(-1)
        ent_label = ent_label.view(-1)
        evt_logit = evt_logit.view(evt_label.size()[0], -1)
        ent_logit = ent_logit.view(ent_label.size()[0], -1)

        loss = 0.33 * self.loss_function(evt_logit, evt_label)
        loss += 0.33 * self.loss_function(ent_logit, ent_label)

        role_label = role_label.view(-1)
        role_logit = role_logit.view(role_label.size()[0], -1)

        loss += 0.34 * self.loss_function(role_logit, role_label)
        return loss
    
    def optimize(self):
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_metric(self, y_pred, y_true, average='micro'):
        evt_pred, ent_pred, role_pred = y_pred
        evt_true, ent_true, role_true = y_true
        evt_p, evt_r, evt_f1 = calc_metrics(evt_true, evt_pred, [i for i in range(1, 2 * len(self.event_list) + 1)], average=average)
        ent_p, ent_r, ent_f1 = calc_metrics(ent_true, ent_pred, [1, 2], average=average)
        role_p, role_r, role_f1 = calc_metrics(role_true, role_pred, [i for i in range(1, len(self.role_list))], average=average)

        precision = (evt_p, ent_p, role_p)
        recall = (evt_r, ent_r, role_r)
        f1 = (evt_f1, ent_f1, role_f1)

        metric = (evt_f1 + ent_f1 + role_f1) / 3
        return precision, recall, f1, metric

    def get_result(self, logits, labels, input_data=None, ignore_index=-100):
        final_preds = tuple()
        final_labels = tuple()
        # evt_logit, ent_logit, role_logit = logits
        # evt_label, ent_label, role_label = labels
        for pred, label in zip(logits, labels):
            label = label.view(-1)
            pred = pred.view(label.size()[0], -1)
            proba = torch.log_softmax(pred, dim=1)
            pred_cls = torch.argmax(proba, dim=1)
            pred_cls = pred_cls.cpu().view(-1).numpy()
            label = label.cpu().view(-1).numpy()

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
        logging.info('  ????????????')
        train_data_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=0)
#         dev_data_loader = torch.utils.data.DataLoader(dataset=self.dev_dataset, batch_size=self.batch_size,
#                                                       shuffle=False, num_workers=0)
        # init
        if self.use_lstm:
            logging.info('  ??????bert??????')
            bert_params = self.model.module.bert.parameters() if self.use_gpu else self.model.bert.parameters()
            for i in bert_params:
                i.requires_grad = False
        if self.retrain:
            checkpoint = self.best_model_save_path
            self.load_model(checkpoint)
            logging.info('  ??????bert??????')
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
                logging.info('  ?????????: ' + str(lrs['lr']))
            # training loop
            bar = tqdm(list(enumerate(train_data_loader)))
            for step, input_data in bar:
                inputs = input_data[:-3]
                labels = input_data[-3:]
                # print(input_data)
                if self.use_gpu:
                    inputs = tuple(x.cuda() for x in inputs)
                    labels = tuple(x.cuda() for x in labels)

                logits = self.model(inputs)
                loss = self.get_loss(logits, labels)
                loss.backward()
                
                # params optimization
                if ((step + 1) % self.accumulate_step) == self.accumulate_step - 1 or step == len(
                        train_data_loader) - 1:
                    self.optimize()

                # calc metric info
                y_pred, y_true = self.get_result(logits, labels)
                precision, recall, f1, metric = self.get_metric(y_pred, y_true)
                bar.set_description(f"step:{step}: pre:{round(precision[0], 3)}, {round(precision[1], 3)}, {round(precision[2], 3)} loss:{loss}")

            # adjust learning rate
            self.scheduler.step()
            # evaluate on dev dataset
#             metric, _ = self.eval(self.model, dev_data_loader)
            metric = self.eval(self.model)
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

        logging.info('  ????????????!\n')
        return True
    
    def calc_standard_metrics(self, pred_path, ground_truth_path):
        pred_items = [json.loads(line) for line in open(pred_path)]
        ground_truth_items = json.load(open(ground_truth_path))
        pred_num = 0
        true_num = 0
        gt_num = 0
        for p, gt in zip(pred_items, ground_truth_items):
            assert p['text'] == gt['text']
            pe = set()
            pa = set()
            for e in p.get("event_list", []):
                e_key = tuple(e['trigger'])
                pe.add(e_key)
                for a in e['argument']:
                    pa.add(tuple(a) + tuple(e_key))
            ge = set()
            ga = set()
            for e in gt.get("event_list", []):
                e_key = (e['event_type'], int(e['beg']), e['event_content'])
                ge.add(e_key)
                for a in e['argument_list']:
                    ga.add((a['argument_role'], int(a['beg']), a['argument_content']) + e_key)
            pred_num += len(pe) + len(pa)
            true_num += len(pe & ge) + len(pa & ga)
            gt_num += len(ge) + len(ga)
        p = float(true_num / pred_num) if pred_num != 0 else 0
        r = float(true_num / gt_num) if gt_num != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        logging.info(f"dev:{self.epoch}: {round(f1, 3)}")
        return f1

    def eval(self, model, mode='dev'):
        self.test(input_path=self.dev_path, output_path=self.eval_output_path, dataset=self.dev_dataset, model=model)
        metrics = self.calc_standard_metrics(self.eval_output_path, self.dev_path)
        return metrics

    def test(self, input_path=None, output_path=None, dataset=None, checkpoint=None, model=None):
        if not output_path:
            output_path = self.test_output_path
        logging.info('  ????????????')
        if not dataset:
            dataset = self.test_dataset
        if not model:
            # load checkpoint
            if not checkpoint:
                checkpoint = self.best_model_save_path
            self.load_model(checkpoint)
        else:
            self.model = model
        # load test dataset
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size,
                                                       shuffle=False, num_workers=0)
        test_result = []
        if not input_path:
            test_contents = [json.loads(line) for line in open(self.test_path)]
        else:
            test_contents = json.load(open(input_path))
        
        offset = 0
        # test
        with torch.no_grad():
            self.model.eval()
            bar = tqdm(list(enumerate(data_loader)))
            for step, input_data in bar:
                # inputs = tuple(x.to(self.device) for x in input_data[:-1])
                inputs = input_data[:2]
                if self.use_gpu:
                    inputs = tuple(x.cuda() for x in inputs)
                    result = self.model.module.predict(inputs)
                else:
                    result = self.model.predict(inputs, use_gpu=False)

                for i in range(len(result)):
                    test_info = test_contents[offset+i]
                    pred_info = self.handle_result(result[i], test_info['text'])
                    
                    pred_info['doc_id'] = test_info.get('doc_id', 0)
                    test_result.append(pred_info)
                
                offset += len(result)
        logging.info("??????test?????????" + output_path)
        
        import jsonlines
        with jsonlines.open(output_path, mode='w') as writer:
            for i in test_result:
#                 if 'text' in i:
#                     del i['text']
                writer.write(i)
#         json.dump(test_result, open(output_path, 'w'), indent=2, ensure_ascii=False)
        return True
    
    def handle_result(self, result, text):
        event_list = []
        if not result:
            return {"text": text, "event_list": []}
        
        args, ents, evts = result
        
        evt_arg_map = {}
        # for beg, end, t in evts:
        #     ent[beg + '@#@' + end] = t
        
        for k in args:
            for j in k:
                evt_ofs, ent_ofs, role_info = j[0], j[1], j[2]
                if role_info == 0 or evt_ofs[-1] == 0:
                    continue
                role = self.role_list[role_info]
                evt_type = self.event_list[evt_ofs[-1]]
                trigger = text[evt_ofs[0]: evt_ofs[1]]
                
                if not trigger:
                    continue
                argument = text[ent_ofs[0]: ent_ofs[1]]
                
                if not argument:
                    continue
                evt_type_key = (evt_type, evt_ofs[0], trigger)
                if evt_type_key not in evt_arg_map:
                    evt_arg_map[evt_type_key] = set()
                # check if evt + role possible
                if evt_type + '@#@' + role not in self.possible_role_list:
                    continue
                evt_arg_map[evt_type_key].add((role, ent_ofs[0], argument))
        
        for et, arg_set in evt_arg_map.items():
            arguments = []
            for arg_info in arg_set:
                arguments.append(list(arg_info))
            event = {
                'trigger': list(et),
                'argument': arguments
            }
            event_list.append(event)

        return {"text": text, "event_list": event_list}

        
        