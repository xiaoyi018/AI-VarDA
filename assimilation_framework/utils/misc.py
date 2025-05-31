import io, os, re
import random

import yaml
import numpy as np
import torch
import torch.distributed as dist

def load_config(config_path):
    with open(config_path, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader = yaml.FullLoader)
    return cfg

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


def get_ip(ip_list):
    if "," in ip_list:
        ip_list = ip_list.split(',')[0]
    if "[" in ip_list:
        ipbefore_4, ip4 = ip_list.split('[')
        ip4 = re.findall(r"\d+", ip4)[0]
        ip1, ip2, ip3 = ipbefore_4.split('-')[-4:-1]
    else:
        ip1,ip2,ip3,ip4 = ip_list.split('-')[-4:]
    ip_addr = "tcp://" + ".".join([ip1, ip2, ip3, ip4]) + ":"
    return ip_addr

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        ip_addr = get_ip(os.environ['SLURM_STEP_NODELIST'])
        port = int(os.environ['SLURM_SRUN_COMM_PORT'])
        # args.init_method = ip_addr + str(port)
        args.init_method = ip_addr + args.init_method.split(":")[-1]
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, local_rank {}): {}'.format(
        args.rank, args.local_rank, args.init_method), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.init_method,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def DistributedParallel_Model(model, gpu_num):
    if is_dist_avail_and_initialized():
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        if device == torch.device('cpu'):
            raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
        model.to(device)
            # model.model[key].to(device)
        print("parallel training initialized")
        ddp_sub_vae  = torch.nn.parallel.DistributedDataParallel(model.vae, device_ids=[gpu_num])
        ddp_sub_fengwu = torch.nn.parallel.DistributedDataParallel(model.fengwu, device_ids=[gpu_num])
        model.vae = ddp_sub_vae
        model.fengwu = ddp_sub_fengwu
        
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    return model

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        # if tensor.is_floating_point():
        #     tensor = nan_to_num(tensor)
        other = tensor.clone()
        # print(name)
        torch.distributed.broadcast(tensor=other, src=0)
        # print(fullname, tensor.sum(), other.sum())
        assert (tensor == other).all(), fullname

class FieldWriter:
    def __init__(self, dest_type, client=None):
        self.dest = dest_type
        self.client = client

    def write_numpy_to_s3(self, data, path):
        with io.BytesIO() as f:
            np.save(f, data)
            f.seek(0)
            self.client.put(path, f)

    def write_numpy_local(self, data, path):
        np.save(path, data)

    def write_numpy(self, data, path):
        if self.dest == "s3":
            self.write_numpy_to_s3(data, path)
        elif self.dest == "local":
            self.write_numpy_local(data, path)
        else:
            raise NotImplementedError
