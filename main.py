import argparse
import sys
import json
import logging
import torch
import random

from common.utils import *
from common.const import *

from module.joint.joint_dataset import *
from module.joint.joint_executor import *
from module.joint.joint_model import *
from module.common.common_dataset import *

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()

# task setting
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--conf', type=str, default='conf/ace05.json')
parser.add_argument('--dataset', type=str, default='duee')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--use_cpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=0)
parser.add_argument('--accumulate_step', type=int, default=0)
parser.add_argument('--task', type=str, default='joint', choices=['joint'])

handler_map = {
    'joint': JointDataHandler
}

model_map = {
    'joint': JointModel
}

dataset_map = {
    'joint': CommonDataSet
}

executor_map = {
    'joint': JointModelExecutor
}

args = parser.parse_args()

def split_dataset(conf):
    source_train_path = conf['source_train_path']
    train_path = conf['train_path']
    dev_path = conf['dev_path']
    data = json.load(open(source_train_path))
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    json.dump(data[:split_idx], open(train_path, 'w'), indent=2, ensure_ascii=False)
    json.dump(data[split_idx:], open(dev_path, 'w'), indent=2, ensure_ascii=False)

def main(args):
    logging.info("-" * 100)
    setup_seed(seed)
    conf = json.load(open(args.conf, encoding='utf-8'))
    
    if args.use_cpu:
        conf['use_gpu'] = 0
    else:
        conf['use_gpu'] = torch.cuda.is_available()
    logging.info("  使用gpu? : " + str(bool(conf['use_gpu'])) + '\n')


    # 更新自定义参数
    if args.epochs:
        conf['epochs'] = args.epochs
    if args.batch_size:
        conf['batch_size'] = args.batch_size
    if args.accumulate_step:
        conf['accumulate_step'] = args.accumulate_step
    
    Handler = handler_map[args.task]
    Model = model_map[args.task]
    Dataset = dataset_map[args.task]
    Executor = executor_map[args.task]
    
    split_dataset(conf)

    mode = args.mode
    if mode == 'train':
        train_dataset = Dataset(args.dataset, Handler, conf.get("train_path"), conf, debug=args.debug)
        # test_dataset = Dataset(args.dataset, EvtTagDataHandler, conf.get("test_path"), conf, debug=args.debug)
        test_dataset = None
        dev_dataset = Dataset(args.dataset, Handler, conf.get("dev_path"), conf, debug=args.debug)
    else:
        conf.update(json.load(open(conf.get('best_model_config_path', "cache/evt_men/best_config.json"))))
        train_dataset = None
        conf['batch_size'] = 64
        test_dataset = Dataset(args.dataset, Handler, conf.get("test_path"), conf, debug=args.debug)
        dev_dataset = Dataset(args.dataset, Handler, conf.get("dev_path"), conf, debug=args.debug)
    
    model = Model(conf)

    model_processor = Executor(
        model,
        train_dataset,
        dev_dataset,
        test_dataset,
        conf
    )

    if mode=='train':
        model_processor.train()
        # model_processor.test()
    else:
        model_processor.test()
        model_processor.test(input_path=conf.get("dev_path"), output_path=conf.get("eval_output_path"), dataset=dev_dataset)
        pass
    return


if __name__ == "__main__":
    logging.info(args)
    main(args)