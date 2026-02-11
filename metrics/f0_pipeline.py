import librosa
import numpy as np
import sys
import os
import pyworld as pw
import torch
import torchaudio
from metrics.pitch import yin, dio
from metrics.utils import same_t_in_true_and_est, gross_pitch_error_frames, voicing_decision_error_frames, true_voiced_frames


try:
    from metrics.rmvpe.model import RMVPEWrapper
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), "metrics", "rmvpe"))
    try:
        from metrics.rmvpe.model import RMVPEWrapper
    except ImportError:
        print("[Warning] Could not import RMVPEWrapper. Ensure metrics/rmvpe/model.py exists.")

class GlobalConfig():
    def __init__(self):

        self.sr = 22050

        self.f0_min_pitch: int = 100
        self.f0_max_pitch: int = 500

        self.n_fft = 1024
        self.win_length: int = 1024
        self.hop_length: int = 256 
        self.f0_min_mel: int = 0
        self.f0_max_mel = None
        self.harmo_thresh: float = 0.1
        self.window_fn =  torch.hann_window
        self.log_mels = True
        self.n_mels = 80
        self.n_mfcc = 13
        self.power = 2
        self.fig_size: tuple = (16,10)
        self.dist_fn = 'compute_rms_dist'
        self.norm_align_type =  'path'

        self.mel_fn = torchaudio.transforms.MelSpectrogram(
                sample_rate = self.sr, n_fft=self.n_fft, win_length=self.win_length,
                hop_length=self.hop_length, f_min=self.f0_min_mel, f_max = self.f0_max_mel,
                n_mels=self.n_mels, window_fn=self.window_fn, power = self.power
            )

        self.melkwargs = {
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "f_min": self.f0_min_mel,
            "f_max": self.f0_max_mel,
            "window_fn": self.window_fn,
            "n_mels": self.n_mels,
            "power": self.power
        }

        self.mfcc_fn = torchaudio.transforms.MFCC(
                sample_rate = self.sr, n_mfcc=self.n_mfcc, log_mels=self.log_mels, melkwargs=self.melkwargs
            )  

@same_t_in_true_and_est
def f0_frame_error(true_t, true_f, est_t, est_f):
    gpe_frames = gross_pitch_error_frames(true_t, true_f, est_t, est_f)
    vde_frames = voicing_decision_error_frames(true_t, true_f, est_t, est_f)
    return (np.sum(gpe_frames) + np.sum(vde_frames)) / (len(true_t))

@same_t_in_true_and_est
def gross_pitch_error(true_t, true_f, est_t, est_f):
    correct_frames = true_voiced_frames(true_t, true_f, est_t, est_f)
    gpe_frames = gross_pitch_error_frames(true_t, true_f, est_t, est_f)
    if np.sum(correct_frames) == 0:
        return 0.0
    return np.sum(gpe_frames) / np.sum(correct_frames)

@same_t_in_true_and_est
def voicing_decision_error(true_t, true_f, est_t, est_f):
    vde_frames = voicing_decision_error_frames(true_t, true_f, est_t, est_f)
    return np.sum(vde_frames) / (len(true_t))

class F0Evaluator:
    # pitch_algorithm: 'pyworld', 'rmvpe', 'dio', 'yin'
    def __init__(self, pitch_algorithm='yin'):
        self.config = GlobalConfig()
        self.pitch_algorithm = pitch_algorithm
        self.rmvpe_model = None
        
        if self.pitch_algorithm == 'rmvpe':
            '''
            if user wants to use rmvpe, make sure to download the rmvpe.pt model from:
            https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt
            and place it in the metrics/rmvpe/ directory.
            '''
            try:
                model_path = os.path.join(os.path.dirname(__file__), "rmvpe", "rmvpe.pt")
                self.rmvpe_model = RMVPEWrapper(model_path)
            except Exception as e:
                print(f"[Error] Failed to initialize RMVPE model: {e}")

    def compute(self, gt_path, synth_path):
        """
        Returns:
            dict: {'ffe': float, 'gpe': float, 'vde': float}
        """
        try:
            dtype = np.float64 if self.pitch_algorithm == 'pyworld' else np.float32
            
            x_gt, sr_gt = librosa.load(gt_path, sr=None, dtype=dtype)
            x_synth, sr_synth = librosa.load(synth_path, sr=None, dtype=dtype)

            if self.pitch_algorithm == 'pyworld':
                target_sr = sr_gt
                if sr_gt != sr_synth:
                    x_gt = librosa.resample(x_gt, orig_sr=sr_gt, target_sr=sr_synth)
                    target_sr = sr_synth
                
                frame_period = 10.0
                f_gt, t_gt = pw.harvest(x_gt, target_sr, frame_period=frame_period)
                f_est, t_est = pw.harvest(x_synth, target_sr, frame_period=frame_period)
                
                min_len = min(len(f_gt), len(f_est))
                f_gt, t_gt = f_gt[:min_len], t_gt[:min_len]
                f_est, t_est = f_est[:min_len], t_est[:min_len]

            elif self.pitch_algorithm == 'rmvpe':
                if self.rmvpe_model is None:
                    print("[Error] RMVPE model not initialized.")
                    return {'ffe': 0.0, 'gpe': 0.0, 'vde': 0.0}

                target_sr = 16000
                if sr_gt != target_sr:
                    x_gt = librosa.resample(x_gt, orig_sr=sr_gt, target_sr=target_sr)
                if sr_synth != target_sr:
                    x_synth = librosa.resample(x_synth, orig_sr=sr_synth, target_sr=target_sr)

                f_gt = self.rmvpe_model.infer(x_gt, thred=0.03)
                f_est = self.rmvpe_model.infer(x_synth, thred=0.03)

                t_gt = np.arange(len(f_gt)) * 0.01
                t_est = np.arange(len(f_est)) * 0.01

            else:
                algo = dio if self.pitch_algorithm == 'dio' else yin

                if sr_gt != sr_synth:
                    x_gt = librosa.resample(x_gt, orig_sr=sr_gt, target_sr=sr_synth)

                pitch_gt = algo(x_gt, self.config)
                pitch_synth = algo(x_synth, self.config)

                t_gt = np.array(pitch_gt['times'])
                f_gt = np.array(pitch_gt['pitches'])
                t_est = np.array(pitch_synth['times'])
                f_est = np.array(pitch_synth['pitches'])

                ffe_score = f0_frame_error(t_gt, f_gt, t_est, f_est)
                gpe_score = gross_pitch_error(t_gt, f_gt, t_est, f_est)

            vde_score = voicing_decision_error(t_gt, f_gt, t_est, f_est)

            return {
                'ffe': float(ffe_score),
                'gpe': float(gpe_score),
                'vde': float(vde_score)
            }
        except Exception as e:
            print(f"F0 Eval Error ({self.pitch_algorithm}): {e}")
            import traceback
            traceback.print_exc()
            return {'ffe': 0.0, 'gpe': 0.0, 'vde': 0.0}