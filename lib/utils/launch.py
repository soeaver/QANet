import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lib.utils.comm import _LOCAL_PROCESS_GROUP, synchronize


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(main_func, num_gpus, args=()):
    port = _find_free_port()
    dist_url = "tcp://127.0.0.1:{}".format(port)

    mp.spawn(
        _distributed_worker,
        nprocs=num_gpus,
        args=(main_func, num_gpus, dist_url, args),
        daemon=False,
    )


def _distributed_worker(
    local_rank, main_func, num_gpus, dist_url, args
):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    try:
        dist.init_process_group(
            backend="NCCL", init_method=dist_url, world_size=num_gpus, rank=local_rank
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    synchronize()
    torch.cuda.set_device(local_rank)

    # Setup the local process group (which contains ranks within the same machine)
    assert _LOCAL_PROCESS_GROUP is None
    ranks = list(range(num_gpus))
    pg = dist.new_group(ranks)
    _LOCAL_PROCESS_GROUP = pg

    main_func(*args)
