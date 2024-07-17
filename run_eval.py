import argparse
from ruamel.yaml import YAML
from pathlib import Path
import datetime
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
from data import get_datasets, vt5_collate_test_fn
from data.captioning import COCO, TextCAPs
from data.documents import DocVQA, InfoVQA
from data.vqa import VizWiz
from data.stvqa import TextVQA
from code_utils import save_result
from accelerate import Accelerator
from get_model import create_model
import os
from functools import partial
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from textvqa_evaluation import TextVQAEvaluator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


global_iter = 0
tokenizer = None
run_name = None
seed = None

@torch.no_grad()
def evaluation(model, data_loader):
    model.eval()
    result = []
    for i, (inputs, question_ids, instructions_list) in tqdm(enumerate(data_loader),
                                                             disable=not accelerator.is_local_main_process):
        generated_ids = model.generate(**inputs, max_new_tokens=30, instructions_list=instructions_list)
        answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for answer, ques_id in zip(answers, question_ids):
            if type(data_loader.dataset) == COCO:
                result.append({"image_id": int(ques_id), "caption": answer.replace('a picture of ','')})
            elif type(data_loader.dataset) == TextCAPs:
                result.append({"image_id": ques_id, "caption": answer.replace('a picture of ','')})
            elif type(data_loader.dataset) in [DocVQA, InfoVQA]:
                result.append({"questionId": ques_id, "answer": answer})
            elif type(data_loader.dataset) == VizWiz:
                result.append({"image": ques_id, "answer": answer})
            else:
                result.append({"question_id": ques_id, "answer": answer})
    return result


def main(args, config):
    # fix the seed for reproducibility
    global tokenizer, evaluator, test_loader, seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False  # Makes it faster if batch size changes

    print_fn("**************************************")
    print_fn("Creating the model")
    model, optimizer, image_processor, tokenizer = create_model(config)

    print_fn(
        f"Number of parameters (total, trainable) : {sum(p.numel() for p in model.parameters())},"
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    #### Checkpoint ####
    if args.ckpt is not None:
        print_fn("**************************************")
        print_fn(f"Loading model from {args.ckpt}")
        state_dict = torch.load(args.ckpt, map_location='cpu')['model']
        m, u = model.load_state_dict(state_dict, strict=False)
        print_fn(f'missing keys: {m}')
        print_fn(f'unexpected keys: {u}')

    #### Dataset ####
    print_fn("**************************************")
    print_fn("Creating datasets")

    _, val_set = get_datasets(config=config, eval=True)
    collate_fn = vt5_collate_test_fn
    collate_test = partial(collate_fn, tokenizer=tokenizer, processor=image_processor)

    test_loader = DataLoader(
        val_set,
        batch_size=config['batch_size_test'],
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_test,
        drop_last=False,
    )
    # Accelerate prepare
    test_loader, optimizer, model = accelerator.prepare(test_loader, optimizer, model)

    print_fn("**************************************")
    print_fn("Starting evaluation")
    # Evaluation if in evaluate mode
    eval_result = evaluation(model, test_loader)
    result_file = save_result(eval_result, args.result_dir, f'result')
    if type(test_loader.dataset) == TextVQA:
        accelerator.wait_for_everyone()
        evaluator = TextVQAEvaluator(dataset_json_file='data/textvqa/train/TextVQA_0.5.1_val.json')
        accuracy = evaluator.evaluate_pred_file(result_file)
        print_fn('accuracy: {:.4f}'.format(accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/textvqa/pretrain_clip_flan_t5_base.yaml')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()
    config_name = os.path.split(args.config)[-1].split('.')[0]
    config_dir = os.path.split(args.config)[-2].split('/')[-1]
    # Meaningful run name for tensorboard
    run_name = config_dir + '_' + config_name + '_' + datetime.datetime.now().strftime("%m_%d_%Y_%H:%M")

    yaml = YAML(typ='rt')
    config = yaml.load(open(args.config, 'r'))

    # Create a unique directory based on parameters
    args.output_dir = os.path.join(args.output_dir, config_dir, config_name, datetime.datetime.now().strftime(
        "%m_%d_%Y_%H:%M"))
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    accelerator = Accelerator()
    print_fn = accelerator.print    # Prints only in the main process

    main(args, config)
