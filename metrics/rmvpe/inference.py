import os
import sys
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
# import matplotlib.pyplot as plt

try:
    from utils import scan_files, get_output_path, load_audio_16k, interpolate_f0
    from model import RMVPEWrapper, Config
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import scan_files, get_output_path, load_audio_16k, interpolate_f0
    from model import RMVPEWrapper, Config

class F0InferenceDataset(Dataset):
    def __init__(self, audio_files, input_root, output_dir, target_sr, hop_size, max_duration):
        self.audio_files = audio_files
        self.input_root = input_root
        self.output_dir = output_dir
        self.target_sr = target_sr
        self.hop_size = hop_size
        self.max_duration = max_duration

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        output_path = get_output_path(audio_path, self.input_root, self.output_dir)
        if not os.path.exists(audio_path):
            return {
                "audio_path": audio_path,
                "output_path": output_path,
                "error": "File not found"
            }
        audio_16k, original_sr, original_len = load_audio_16k(audio_path)
        return {
            "audio_path": audio_path,
            "output_path": output_path,
            "audio_16k": audio_16k,
            "original_sr": original_sr,
            "original_len": original_len
        }

def parse_arguments():
    parser = argparse.ArgumentParser(description="RMVPE F0 DDP distributed inference tool")
    parser.add_argument("-i", "--input", required=True, help="Input path (folder or single audio file)")
    parser.add_argument("-o", "--output", default=None, help="Output folder path")
    default_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rmvpe.pt")
    parser.add_argument("-m", "--model", default=default_model, help="RMVPE model path")
    parser.add_argument("--ngpu", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--target_sr", type=int, default=24000, help="Target sample rate")
    parser.add_argument("--hop_size", type=int, default=480, help="Hop size")
    parser.add_argument("--max_duration", type=float, default=20.0, help="Max duration (seconds)")
    return parser.parse_args()

def create_config_from_args(args):
    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output) if args.output else None
    model_path = os.path.abspath(args.model)
    devices = [f"cuda:{i}" for i in range(args.ngpu)]
    config = Config(
        input_path=input_path,
        output_dir=output_dir,
        model_path=model_path,
        devices=devices,
        workers=args.num_workers,
        target_sr=args.target_sr,
        hop_size=args.hop_size,
        max_duration=args.max_duration
    )
    return config

def merge_result_files(output_dir, world_size):
    print("Merging result files...")
    merged = 0
    with open(os.path.join(output_dir, "f0_ddp_merged.txt"), "w") as fout:
        for rank in range(world_size):
            result_path = os.path.join(output_dir, f"f0_ddp_rank{rank}.txt")
            if not os.path.exists(result_path):
                continue
            with open(result_path, "r") as fin:
                for line in fin:
                    fout.write(line)
                    merged += 1
            os.remove(result_path)
    print(f"Merged {merged} results to {os.path.join(output_dir, 'f0_ddp_merged.txt')}")

def ddp_worker(rank, world_size, config, audio_files):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"[DDP] Rank {rank} started, device cuda:{rank}")

    dataset = F0InferenceDataset(
        audio_files,
        config.input_path,
        config.output_dir,
        config.target_sr,
        config.hop_size,
        config.max_duration
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=config.workers, pin_memory=True)

    model = RMVPEWrapper(config.model_path, device=f"cuda:{rank}")

    result_path = os.path.join(config.output_dir, f"f0_ddp_rank{rank}.txt")
    with open(result_path, "w") as fout:
        for batch in tqdm(loader, desc=f"Rank {rank}"):
            batch = batch[0] if isinstance(batch, list) else batch
            if "error" in batch:
                fout.write(f"{batch['audio_path']}\t{batch['error']}\n")
                continue
            try:
                batch["audio_16k"] = batch["audio_16k"].squeeze(0)
                f0_16k = model.infer(batch["audio_16k"], thred=0.03)
                f0_final = interpolate_f0(
                    f0_16k,
                    batch["original_len"],
                    batch["original_sr"],
                    target_sr=config.target_sr,
                    hop_size=config.hop_size,
                    max_duration=config.max_duration
                )
                np.save(batch["output_path"][0], f0_final)
                fout.write(f"{batch['audio_path'][0]}\t{batch['output_path'][0]}\n")
            except Exception as e:
                fout.write(f"{batch['audio_path']}\tError: {str(e)}\n")
    dist.destroy_process_group()

def main_distributed():
    args = parse_arguments()
    config = create_config_from_args(args)
    if not os.path.exists(config.input_path):
        print(f"not exist: {config.input_path}")
        sys.exit(1)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir, exist_ok=True)
    audio_files = scan_files(config.input_path)
    if not audio_files:
        print("not found any audio files")
        sys.exit(0)
    print(f"found {len(audio_files)} audio files")
    world_size = len(config.devices)
    mp.spawn(ddp_worker, args=(world_size, config, audio_files), nprocs=world_size, join=True)
    merge_result_files(config.output_dir, world_size)

if __name__ == "__main__":
    main_distributed()