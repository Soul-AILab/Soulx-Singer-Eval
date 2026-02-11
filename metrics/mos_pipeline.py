import os
import sys
import torch
import torchaudio
import librosa
import numpy as np
import yaml
import torch.nn.functional as F

class MOSEvaluator:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
        self.ckpt_dir = os.path.join(self.root_dir, "ckpt")

        # https://github.com/unilight/sheet/releases/download/v0.1.0/all7-sslmos-mdf-2337-config.yml
        self.sheet_config = os.path.join(self.ckpt_dir, "all7-sslmos-mdf-2337-config.yml")
        # https://github.com/unilight/sheet/releases/download/v0.1.0/all7-sslmos-mdf-2337-checkpoint-86000steps.pkl
        self.sheet_ckpt = os.path.join(self.ckpt_dir, "all7-sslmos-mdf-2337-checkpoint-86000steps.pkl")
        # ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth
        self.singmos_ckpt = os.path.join(self.ckpt_dir, "ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth")

        print("Loading MOS Models...")
        self.singmos = self._load_singmos()
        self.sheet = self._load_sheet()

    def _load_singmos(self):
        try:
            from metrics.singmos.singmos.ssl_mos.singmos_pro import MOS_Predictor
            model = MOS_Predictor(ssl_model_type="wav2vec2_large_ll60k", use_domain_id=True, domain_num=6)
            state_dict = torch.load(self.singmos_ckpt, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            return model.to(self.device)
        except Exception as e:
            print(f"Warning: Failed to load SingMOS: {e}")
            return None

    def _load_sheet(self):
        try:
            from sheet.models.sslmos import SSLMOS
            with open(self.sheet_config, 'r') as f:
                config = yaml.load(f, Loader=yaml.Loader)
            model = SSLMOS(config["model_input"], **config["model_params"])
            state_dict = torch.load(self.sheet_ckpt, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            return SheetPredictor(model.to(self.device), config, self.device)
        except Exception as e:
            print(f"Warning: Failed to load Sheet-SSQA: {e}")
            return None

    def compute_all(self, wav_path):
        results = {}
        
        # 1. SingMOS
        if self.singmos:
            try:
                wav, sr = librosa.load(wav_path, sr=16000, mono=True)
                wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(self.device)
                length = torch.tensor([wav_tensor.shape[1]], dtype=torch.long).to(self.device)
                with torch.no_grad():
                    results['singmos'] = self.singmos(wav_tensor, length).item()
            except:
                results['singmos'] = 0.0
        else:
            results['singmos'] = 0.0

        # 2. Sheet-SSQA
        if self.sheet:
            try:
                results['sheet'] = self.sheet.predict(wav_path)
            except:
                results['sheet'] = 0.0
        else:
            results['sheet'] = 0.0

        return results

class SheetPredictor:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.fs = 16000
        self.device = device

    def predict(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        if sr != self.fs:
            resampler = torchaudio.transforms.Resample(sr, self.fs)
            wav = resampler(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.shape[1] < 1040:
             pad = 1040 - wav.shape[1]
             wav = F.pad(wav, (0, pad))

        model_input = wav
        model_lengths = torch.tensor([model_input.size(1)]).long()
        inputs = {
            self.config["model_input"]: model_input.to(self.device),
            self.config["model_input"] + "_lengths": model_lengths.to(self.device),
        }

        with torch.no_grad():
            if self.config["inference_mode"] == "mean_listener":
                outputs = self.model.mean_listener_inference(inputs)
            else:
                outputs = self.model.mean_net_inference(inputs)
        return outputs["scores"].cpu().mean().item()