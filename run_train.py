import argparse
from pathlib import Path
import datetime
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
from ruamel.yaml import YAML

from data import get_datasets, vt5_collate_test_fn, vt5_collate_train_fn
from code_utils import save_result, linear_lr_schedule, cosine_lr_schedule, warmup_lr_schedule, const_lr_schedule, \
    copy_codebase, to_model_device
from textvqa_evaluation import TextVQAEvaluator
from get_model import create_model
from accelerate import Accelerator
import os
from functools import partial
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


global_iter = 0
tokenizer = None
run_name = None
translation_mul = None
evaluator = None
eval_loader = None
total_training_steps = None
best_acc = None
training_stats = None


def train(model, data_loader, optimizer, epoch):
    model.train()
    print_fn("**************************************")
    print_fn(f'Train epoch: [{epoch}]')
    eval_freq, print_freq = 1000, 100

    optimizer.zero_grad()
    avg_loss = 0
    global global_iter, translation_mul, evaluator, eval_loader, total_training_steps, best_acc, training_stats
    with tqdm(enumerate(data_loader), disable=not accelerator.is_local_main_process, unit="batch") as tepoch:
        acc_steps = accelerator.gradient_accumulation_steps
        for i, inputs in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            # Apply warmup
            if config['warmup_iters'] > global_iter // acc_steps:
                warmup_lr_schedule(optimizer, global_iter // acc_steps, config['warmup_iters'], 1e-6, config['base_lr'],
                                   multiplier=translation_mul)
            else:   # LR scheduling
                if config['scheduler'] == 'linear':
                    linear_lr_schedule(optimizer, global_iter, total_training_steps, config['base_lr'], config['min_lr'],
                                       multiplier=translation_mul)
                elif config['scheduler'] == 'cosine':
                    cosine_lr_schedule(optimizer, global_iter, total_training_steps, config['base_lr'], config['min_lr'],
                                       multiplier=translation_mul)
                elif config['scheduler'] == 'constant':
                    const_lr_schedule(optimizer, global_iter, total_training_steps, config['base_lr'], config['min_lr'],
                                      multiplier=translation_mul)
            # Accelerate handles the grad accumulation, if applicable
            with accelerator.accumulate(model):
                inputs = to_model_device(inputs, device=model.device, dtype=model.base_model.dtype)
                loss = model(**inputs).loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                avg_loss += accelerator.gather(loss).mean().item()
                if len(optimizer.param_groups) > 1:
                    tepoch.set_postfix(loss=accelerator.gather(loss).mean().item(),
                                       g1_lr=optimizer.param_groups[0]['lr'],
                                       g2_lr=optimizer.param_groups[1]['lr'])
                else:
                    tepoch.set_postfix(loss=accelerator.gather(loss).mean().item(),
                                       g1_lr=optimizer.param_groups[0]['lr'])

                if global_iter % (print_freq * acc_steps) == 0:
                    print_fn(f'\nglobal iter {global_iter // acc_steps}, average loss {avg_loss / (i + 1)}')
                if global_iter % (eval_freq * acc_steps) == 0 and global_iter > 0:
                    textvqa_result = evaluation(model, eval_loader, epoch)
                    result_file = save_result(textvqa_result, args.result_dir, f'result_{global_iter // acc_steps}')
                    accelerator.wait_for_everyone()
                    accuracy = evaluator.evaluate_pred_file(result_file)
                    print_fn('accuracy: {:.4f}'.format(accuracy))
                    training_stats.append(f'Evaluating global iter {global_iter // acc_steps}, accuracy {accuracy}\n')
                    #
                    if accuracy > best_acc:
                        print_fn(f'\nAccuracy improved from {best_acc} to {accuracy} in global iter'
                                 f' {global_iter // acc_steps}')
                        best_acc = accuracy
                        accelerator.save({
                            'model': accelerator.unwrap_model(model).state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'config': config,
                            'epoch': epoch,
                        }, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    model.train()   # To return to training mode after intermediate evaluation
                global_iter += 1
    return avg_loss / i


@torch.no_grad()
def evaluation(model, data_loader, epoch):
    print_fn("**************************************")
    print_fn(f'Evaluate epoch: [{epoch}]')
    model.eval()
    result = []
    for i, (inputs, question_ids, instructions_list) in tqdm(enumerate(data_loader),
                                                             disable=not accelerator.is_local_main_process):
        generated_ids = model.generate(**inputs, max_new_tokens=30, instructions_list=instructions_list)
        answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for answer, ques_id in zip(answers, question_ids):
            result.append({"question_id": ques_id, "answer": answer})
    return result


def main(args, config):
    # fix the seed for reproducibility
    global tokenizer, val_json, evaluator, eval_loader, image_processor, best_acc, total_training_steps, \
        translation_mul, training_stats
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    cudnn.benchmark = False

    #### Model ####
    model, optimizer, image_processor, tokenizer = create_model(config)
    print_fn(
        f"Number of parameters (total, trainable) : {sum(p.numel() for p in model.parameters())},"
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    unwrapped_model = accelerator.unwrap_model(model)
    translation_mul = config['translation_lr_mul'] if 'translation_lr_mul' in config else 1.

    #### Dataset ####
    print_fn("Creating datasets")
    print_fn("**************************************")
    train_set, val_set = get_datasets(config=config)

    collate_test_fn = vt5_collate_test_fn
    collate_train_fn = vt5_collate_train_fn
    collate_test = partial(collate_test_fn, tokenizer=tokenizer, processor=image_processor)
    collate_train = partial(collate_train_fn, tokenizer=tokenizer, processor=image_processor)

    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size_train'],
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_train,
        drop_last=False,
    )
    eval_loader = DataLoader(
        val_set,
        batch_size=config['batch_size_test'],
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_test,
        drop_last=False,
    )
    # Accelerate prepare
    train_loader, eval_loader, model, optimizer = accelerator.prepare(train_loader, eval_loader, model, optimizer)

    evaluator = TextVQAEvaluator(dataset_json_file='data/textvqa/train/TextVQA_0.5.1_val.json')

    print_fn(f'Training stats: epoch {config["n_epochs"]}, data_size {train_loader.total_dataset_length}, '
             f'effective batch_size {accelerator.gradient_accumulation_steps * accelerator.num_processes * config["batch_size_train"]}')

    print_fn("Start training")
    print_fn("**************************************")
    best_acc = -1
    training_stats = []
    total_training_steps = config['n_epochs'] * (train_loader.total_dataset_length / (
            accelerator.gradient_accumulation_steps * accelerator.num_processes * config["batch_size_train"]))

    for epoch in range(0, config['n_epochs']):
        avg_loss = train(model, train_loader, optimizer, epoch)
        training_stats.append(f'Training epoch {epoch}, average loss {avg_loss}\n')
        # Evaluation after epoch
        textvqa_result = evaluation(model, eval_loader, epoch)
        result_file = save_result(textvqa_result, args.result_dir, f'result_epoch_{epoch}')
        accelerator.wait_for_everyone()
        accuracy = evaluator.evaluate_pred_file(result_file)
        print_fn(f'accuracy: {accuracy}')
        training_stats.append(f'Evaluating epoch {epoch}, accuracy {accuracy}\n')
        if accuracy > best_acc:
            best_epoch = epoch
            print_fn(f'Accuracy improved from {best_acc} to {accuracy} in epoch {best_epoch}')
            best_acc = accuracy
            # #
            accelerator.save({
                'model': unwrapped_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }, os.path.join(args.output_dir, 'checkpoint_best.pth'))

    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, 'training_log.txt'), 'a') as f:
            f.writelines(training_stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/VisualT5_base/qavit.yaml', type=str)
    parser.add_argument('--output_dir', default='output', type=str)
    parser.add_argument('--seed', default=None, type=int)
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

    # Backup the codebase to ensure reproducibility
    copy_codebase(args.output_dir)
    accelerator = Accelerator()
    print_fn = accelerator.print    # Prints only in the main process

    main(args, config)