import argparse
import sys
import json
import logging
import torch
sys.path.append("../sentence_evt_ext/")
sys.path.append("./")
from common.utils import *
from common.const import *
from sub_module.common.common_seq_tag_dataset import CommonDataSet
from sub_module.art_joint.art_joint_dataset import *
from sub_module.joint.joint_executor import *
from sub_module.art_joint.art_joint_model import *

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
parser.add_argument('--task', type=str, default='evt_men_detect', choices=['joint'])

handler_map = {
    'joint': ArtJointDataHandler
}

model_map = {
    'joint': ArtJointModel
}

dataset_map = {
    'joint': CommonDataSet
}

executor_map = {
    'joint': JointModelExecutor
}

args = parser.parse_args()

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
    
    mode = args.mode
    if mode == 'train':
        train_dataset = Dataset(args.dataset, Handler, conf.get("train_path"), conf, debug=args.debug)
        # test_dataset = Dataset(args.dataset, EvtTagDataHandler, conf.get("test_path"), conf, debug=args.debug)
        test_dataset = None
        dev_dataset = Dataset(args.dataset, Handler, conf.get("dev_path"), conf, debug=args.debug)
    else:
        conf.update(json.load(open(conf.get('best_model_config_path', "cache/evt_men/best_config.json"))))
        train_dataset = None
        dev_dataset = None
        test_dataset = Dataset(args.dataset, Handler, conf.get("test_path"), conf, debug=args.debug)
    
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
        pass
    return


if __name__ == "__main__":
    logging.info(args)
    main(args)