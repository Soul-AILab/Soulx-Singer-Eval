import os
import numpy as np
import soundfile as sf
import librosa
import math
from scipy.interpolate import interp1d

def scan_files(input_path, extensions=('.wav', '.flac', '.mp3', '.ogg', '.m4a')):
    """
    Recursively scan for audio files.
    """
    files = []
    if os.path.isfile(input_path):
        files.append(input_path)
    elif os.path.isdir(input_path):
        for root, _, filenames in os.walk(input_path):
            for f in filenames:
                if f.lower().endswith(extensions):
                    files.append(os.path.join(root, f))
    return files

def get_output_path(file_path, input_root, output_dir):
    """
    Calculate output file path, keeping relative directory structure.
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    filename = f"{base_name}_f0.npy"
    
    if output_dir:
        if os.path.isdir(input_root):
            rel_path = os.path.relpath(os.path.dirname(file_path), input_root)
            target_dir = os.path.join(output_dir, rel_path)
        else:
            target_dir = output_dir
            
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, filename)
    else:
        return os.path.join(os.path.dirname(file_path), filename)

def load_audio_16k(audio_path):
    """
    Read audio and forcibly convert to 16k mono. Also return original info.
    """
    try:
        audio, sr = sf.read(audio_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read audio: {e}")

    original_sr = sr
    original_length = len(audio)
    
    # Convert to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # RMVPE requires 16k input
    if sr != 16000:
        audio_16k = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio
        
    return audio_16k, original_sr, original_length

def interpolate_f0(f0_16k, original_length, original_sr, target_sr=48000, hop_size=256, max_duration=20.0):
    mel_target_sr = target_sr
    mel_hop_size = hop_size
    mel_max_duration = max_duration
    
    batch_max_length = int(mel_max_duration * mel_target_sr / mel_hop_size) 
    duration_in_seconds = original_length / original_sr
    effective_48k_length = int(duration_in_seconds * mel_target_sr)
    original_frames = math.ceil(effective_48k_length / mel_hop_size)
    target_frames = min(original_frames, batch_max_length)

    rmvpe_hop = 160
    
    t_16k = np.arange(len(f0_16k)) * (rmvpe_hop / 16000.0)
    t_target = np.arange(target_frames) * (mel_hop_size / float(mel_target_sr))
    
    if len(f0_16k) > 0:
        f_interp = interp1d(
            t_16k, f0_16k, 
            kind='linear', 
            bounds_error=False, 
            fill_value=0.0,
            assume_sorted=True
        )
        f0 = f_interp(t_target)
    else:
        f0 = np.zeros(target_frames)
    
    if len(f0) != target_frames:
        f0 = f0[:target_frames] if len(f0) > target_frames else \
             np.pad(f0, (0, target_frames - len(f0)), 'constant')
             
    return f0