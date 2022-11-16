"""Guard for running certain operations on main process only

Authors:
 * Abdel Heba 2020
 * Aku Rouhe 2020
"""
import os
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset

from scaler_gan.scalergan_utils.global_logger import logger

rank = 0


def run_on_main(
    func: Callable,
    args: Optional[List[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    post_func: Optional[Callable] = None,
    post_args: Optional[List[Any]] = None,
    post_kwargs: Optional[Dict[str, Any]] = None,
    run_post_on_main: Optional[bool] = False,
):
    """
    Runs a function with DPP (multi-gpu) support.

    The main function is only run on the main process.
    A post_function can be specified, to be on non-main processes after the main
    func completes. This way whatever the main func produces can be loaded on
    the other processes.
    :param func: Function to run on the main process.
    :param args: Positional args to pass to func.
    :param kwargs: Keyword args to pass to func.
    :param post_func: Function to run after func has finished on main. By default only run on
        non-main processes.
    :param post_args: Positional args to pass to post_func.
    :param post_kwargs: Keyword args to pass to post_func.
    :param run_post_on_main: Whether to run post_func on main process as well. (default: False)
    :return:
    """

    # Handle the mutable data types' default args:
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    if post_args is None:
        post_args = []
    if post_kwargs is None:
        post_kwargs = {}

    if if_main_process():
        # Main comes here
        try:
            func(*args, **kwargs)
        finally:
            ddp_barrier()
    else:
        # Others go here
        ddp_barrier()
    if post_func is not None:
        if run_post_on_main:
            # Just run on every process without any barrier.
            post_func(*post_args, **post_kwargs)
        elif not if_main_process():
            # Others go here
            try:
                post_func(*post_args, **post_kwargs)
            finally:
                ddp_barrier()
        else:
            # But main comes here
            ddp_barrier()


def if_main_process() -> bool:
    """
    Checks if the current process is the main process and authorized to run
    I/O commands. In DDP mode, the main process is the one with RANK == 0.
    In standard mode, the process will not have `RANK` Unix var and will be
    authorized to run the I/O commands.
    :return: True if the current process is the main process otherwise return False
    """

    if "RANK" in os.environ:
        if os.environ["RANK"] == "":
            return False
        else:
            if int(os.environ["RANK"]) == 0:
                return True
            return False
    return True


def ddp_barrier() -> None:
    """In DDP mode, this function will synchronize all processes.
    torch.distributed.barrier() will block processes until the whole
    group enters this function.
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def ddp_init_group(
    distributed: bool,
    distributed_backend: str = "nccl",
    local_rank: Optional[int] = None,
):
    """
    This function will initialize the ddp group if
    distributed=True bool is given in the python command line.

    The ddp group will use distributed_backend arg for setting the
    DDP communication protocol. `RANK` Unix variable will be used for
    registering the subprocess to the ddp group.
    :param distributed: A boolean flag for distributed launch or not
    :param distributed_backend: A distributed backend to use default is nccl (if distributed is True).
    :param local_rank: An index of the rank default is None
    :return:
    """

    if distributed:
        if local_rank is None:
            raise ValueError(
                "To use DDP backend, start your script with:\n\t"
                "python -m torch.distributed.launch [args]\n\t"
                "train_mel.py --distributed --distributed_backend=nccl"
            )
        else:
            if local_rank + 1 > torch.cuda.device_count():
                raise ValueError(
                    "Killing process " + str() + "\n" "Not enough GPUs available!"
                )
        if "RANK" in os.environ is None or os.environ["RANK"] == "":
            raise ValueError(
                "To use DDP backend, start your script with:\n\t"
                "python -m torch.distributed.launch [args]\n\t"
                "experiment.py hyperparams.yaml --distributed=True "
                "--distributed_backend=nccl"
            )
        global rank
        rank = int(os.environ["RANK"])

        if distributed_backend == "nccl":
            if not torch.distributed.is_nccl_available():
                raise ValueError("NCCL is not supported in your machine.")
        elif distributed_backend == "gloo":
            if not torch.distributed.is_gloo_available():
                raise ValueError("GLOO is not supported in your machine.")
        elif distributed_backend == "mpi":
            if not torch.distributed.is_mpi_available():
                raise ValueError("MPI is not supported in your machine.")
        else:
            logger.info(distributed_backend + " communcation protocol doesn't exist.")
            raise ValueError(
                distributed_backend + " communcation protocol doesn't exist."
            )
        # rank arg is used to set the right rank of the current process for ddp.
        # if you have 2 servers with 2 gpu:
        # server1:
        #   GPU0: local_rank=device=0, rank=0
        #   GPU1: local_rank=device=1, rank=1
        # server2:
        #   GPU0: local_rank=device=0, rank=2
        #   GPU1: local_rank=device=1, rank=3
        torch.distributed.init_process_group(backend=distributed_backend, rank=rank)
    else:
        logger.info(
            "distributed flag is disabled, "
            "this experiment will be executed without DDP."
        )
        if local_rank is not None and local_rank > 0:
            raise ValueError(
                "DDP is disabled, local_rank must not be set.\n"
                "For DDP training, please use --distributed=True. "
                "For example:\n\tpython -m torch.distributed.launch "
                "experiment.py hyperparams.yaml "
                "--distributed=True --distributed_backend=nccl"
            )


def loader(
    dataset, *args, shuffle=False, dataloader_class=DataLoader, is_ddp=False, **kwargs
):
    """loader.

    Create a dataloader properly in case of distributed training.
    If a gradient is going to be computed you must set `shuffle=True`.

    :param dataset: the dataset to be parallelized
    :param args: relevant args for the loader
    :param shuffle: shuffle examples
    :param dataloader_class: loader class
    :param kwargs: relevant args
    """

    # if not is_ddp:
    #     return dataloader_class(dataset, *args, shuffle=shuffle, **kwargs)

    if shuffle:
        # train means we will compute backward, we use DistributedSampler
        sampler = DistributedSampler(dataset)
        # We ignore shuffle, DistributedSampler already shuffles
        return dataloader_class(dataset, *args, **kwargs, sampler=sampler)
    else:
        # We make a manual shard, as DistributedSampler otherwise replicate some examples
        dataset = Subset(
            dataset, list(range(rank, len(dataset), torch.cuda.device_count()))
        )
        return dataloader_class(dataset, *args, shuffle=shuffle)
