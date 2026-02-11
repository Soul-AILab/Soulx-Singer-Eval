import os
import torch
import librosa
import torch.nn.functional as F
from modelscope.pipelines import pipeline
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


class SVPipeline:
    
    def __init__(self, model_path_or_id="microsoft/wavlm-base-plus-sv", device='cuda'):
        try:
            print(f"Loading WavLM from: {model_path_or_id}")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path_or_id)
            self.sv_model = WavLMForXVector.from_pretrained(model_path_or_id)
        except Exception as e:
            raise RuntimeError(f"Failed to load WavLM from '{model_path_or_id}': {e}")
        
        self.sv_model = self.sv_model.to(device)


    def compute_cos_sim_score(self, wav_1, wav_2):
        spk1_wav, _ = librosa.load(wav_1, sr=16000)
        spk2_wav, _ = librosa.load(wav_2, sr=16000)

        inputs_1 = self.feature_extractor(
            [spk1_wav], padding=True, return_tensors="pt", sampling_rate=16000
        )
        if torch.cuda.is_available():
            for key in inputs_1.keys():
                inputs_1[key] = inputs_1[key].to(self.sv_model.device)
        with torch.no_grad():
            embds_1 = self.sv_model(**inputs_1).embeddings
            embds_1 = embds_1[0]

        inputs_2 = self.feature_extractor(
            [spk2_wav], padding=True, return_tensors="pt", sampling_rate=16000
        )
        if torch.cuda.is_available():
            for key in inputs_2.keys():
                inputs_2[key] = inputs_2[key].to(self.sv_model.device)

        with torch.no_grad():
            embds_2 = self.sv_model(**inputs_2).embeddings
            embds_2 = embds_2[0]
        cos_sim = F.cosine_similarity(embds_1, embds_2, dim=-1).item()
          
        return cos_sim
