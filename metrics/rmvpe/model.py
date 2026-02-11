import sys
import os
import torch
import queue
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Config:
    # Path configuration
    input_path: str
    output_dir: Optional[str]
    model_path: str
    
    # Device configuration
    devices: List[str]
    workers: int
    
    # Audio processing configuration
    target_sr: int = 48000
    hop_size: int = 256
    max_duration: float = 20.0


class RMVPEWrapper:
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        self._init_model()

    def _init_model(self):
        try:
            from metrics.rmvpe.rmvpe_local import RMVPE
        except ImportError as e:
            raise ImportError(
                f"Failed to import RMVPE: {e}.\n"
                f"Please make sure rmvpe_local.py exists in the current directory."
            )
        
        try:
            # print(f"[Info] Loading RMVPE model from {self.model_path} to {self.device}...")
            self.model = RMVPE(
                self.model_path,
                is_half=True if 'cuda' in self.device else False,
                device=self.device
            )
            # print(f"[Info] Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {e}")

    def infer(self, audio_16k, thred=0.03):
        """
        Run inference.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        with torch.no_grad():
            f0 = self.model.infer_from_audio(audio_16k, thred=thred)
        return f0

