import math
import json
from pathlib import Path
import shutil
import os
import torch
import torch.distributed as dist


def to_model_device(inputs, dtype, device):
    for key, value in inputs.items():
        if key in ['pixel_values', 'images']:
            inputs[key] = inputs[key].to(device, dtype=dtype)
        else:
            if type(inputs[key]) != list and inputs[key] is not None:
                inputs[key] = inputs[key].to(device)
    return inputs


def copy_codebase(exp_dir):
    backup_directory = os.path.join(exp_dir, 'code')
    Path(backup_directory).mkdir(parents=True, exist_ok=True)
    dirs_to_backup = ['.', './models', './data']
    # Copy only .py files from the source directory to the backup directory
    for source_directory in dirs_to_backup:
        for filename in os.listdir(source_directory):
            if filename.endswith(".py"):
                source_file = os.path.join(source_directory, filename)
                backup_dir = os.path.join(backup_directory, source_directory)
                Path(backup_dir).mkdir(parents=True, exist_ok=True)
                backup_file = os.path.join(backup_dir, filename)
                if filename.endswith(".py") and os.path.isfile(source_file):
                    shutil.copy2(source_file, backup_file)

    print("Code backup (only .py files) completed.")


def const_lr_schedule(**kwargs):
    """Decay the learning rate"""
    return


def multistep_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr, multiplier=1, alpha=0.1):
    assert len(optimizer.param_groups) <= 2
    lr = init_lr
    if epoch >= max_epoch // 2:
        lr *= alpha
    if epoch >= (3 * max_epoch) // 4:
        lr *= alpha
    print(f"Setting base LR to {lr}")
    for param_group_id, param_group in enumerate(optimizer.param_groups):
        if param_group_id > 0:
            param_group['lr'] = lr * multiplier
        else:
            param_group['lr'] = lr


def linear_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr, multiplier=1):
    """Decay the learning rate"""
    assert len(optimizer.param_groups) <= 2
    lr = (init_lr - min_lr) * (1 - (epoch / max_epoch)) + min_lr
    for param_group_id, param_group in enumerate(optimizer.param_groups):
        if param_group_id > 0:
            param_group['lr'] = lr * multiplier
        else:
            param_group['lr'] = lr


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr, multiplier=1):
    """Decay the learning rate"""
    assert len(optimizer.param_groups) <= 2
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    # print(f'setting lr to {lr}, {epoch}, {max_epoch}, {init_lr, min_lr}')
    for param_group_id, param_group in enumerate(optimizer.param_groups):
        if param_group_id > 0:
            param_group['lr'] = lr * multiplier
        else:
            param_group['lr'] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr, multiplier=1):
    assert len(optimizer.param_groups) <= 2
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group_id, param_group in enumerate(optimizer.param_groups):
        if param_group_id > 0:
            param_group['lr'] = lr * multiplier
        else:
            param_group['lr'] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate, multiplier):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate ** epoch))
    for param_group_id, param_group in enumerate(optimizer.param_groups):
        if param_group_id > 0:
            param_group['lr'] = lr * multiplier
        else:
            param_group['lr'] = lr


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json' % (filename, get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json' % filename)

    json.dump(result, open(result_file, 'w'))

    dist.barrier()

    if is_main_process():
        # combine results from all processes
        result = []

        for rank in range(get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json' % (filename, rank))
            res = json.load(open(result_file, 'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        json.dump(result, open(final_result_file, 'w'))
        print('result file saved to %s' % final_result_file)

    return final_result_file
