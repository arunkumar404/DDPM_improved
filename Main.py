import os
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from Diffusion.Train import train, eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def set_nccl_env_vars():
    os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker0"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"


def main_worker(rank, world_size, model_config):
    logging.info(f"Starting main_worker on rank {rank}")
    set_nccl_env_vars()
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)
    model_config["device"] = rank
    logging.info(f"Initialized process group for rank {rank}")
    if model_config["state"] == "train":
        logging.info(f"Starting training for rank {rank}")
        train(rank, world_size, model_config)
    else:
        logging.info(f"Starting evaluation for rank {rank}")
        eval(rank, world_size, model_config)
    dist.destroy_process_group()
    logging.info(f"Finished main_worker on rank {rank}")


def main(model_config=None):
    logging.info("Starting main function")
    modelConfig = {
        "state": "train",  # or eval
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.0,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.0,
        "device": "cuda", 
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8,
    }
    if model_config is not None:
        modelConfig = model_config

    world_size = torch.cuda.device_count()
    logging.info(f"Spawning {world_size} processes for distributed training")
    mp.spawn(main_worker, args=(world_size, modelConfig), nprocs=world_size, join=True)
    logging.info("Main function completed")


if __name__ == "__main__":
    main()
