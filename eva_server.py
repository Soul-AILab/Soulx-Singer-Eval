import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import torch

from metrics.sv_pipeline import SVPipeline
from metrics.asr_pipeline import ASRPipeline
from metrics.mcd import extract_mcd
from metrics.mos_pipeline import MOSEvaluator
from metrics.f0_pipeline import F0Evaluator

app = FastAPI()

models = {}

class EvalRequest(BaseModel):
    ref_text: str
    ref_wav: str
    gen_wav: str
    prompt_wav: Optional[str] = None
    lang: str = "zh"  # zh or en

print("=== [Server] Loading Models... ===")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Loading ASR Models (ZH & EN)...")
models['asr_zh'] = ASRPipeline(lang='zh')
models['asr_en'] = ASRPipeline(lang='en')

print("Loading SV Model...")
models['sv'] = SVPipeline(model_path_or_id="microsoft/wavlm-base-plus-sv", device=device)

print("Loading Integrated MOS Evaluator...")
models['mos'] = MOSEvaluator(device=device)

print("Loading F0 Evaluator")
models['f0'] = F0Evaluator()

print("=== [Server] Ready! ===")

@app.post("/evaluate")
async def evaluate_sample(req: EvalRequest):
    result_dict = {'gen_wav': req.gen_wav}
    
    # === 1. MCD (Mel Cepstral Distortion) ===
    try:
        result_dict['mcd'] = extract_mcd(req.ref_wav, req.gen_wav)
    except:
        result_dict['mcd'] = 0.0

    # === 2. Speaker Similarity (Cos Sim) ===
    try:
        sv_model = models['sv']
        result_dict['prompt_gen_cos_sim'] = sv_model.compute_cos_sim_score(req.prompt_wav, req.gen_wav)
    except Exception as e:
        print(f"[SV Error] {e}")
        result_dict['prompt_gen_cos_sim'] = 0.0
        
    # === 3. ASR (WER/CER) ===
    try:
        lang_key = 'zh' if req.lang == 'zh' else 'en'
        asr_model = models[f'asr_{lang_key}']
        
        if lang_key == 'zh':
            hyp_text = asr_model.infer_zh(req.gen_wav)
        else:
            hyp_text = asr_model.infer_en(req.gen_wav)
            
        wer_res = asr_model.get_wer(req.ref_text, hyp_text, mode="wer")
        cer_res = asr_model.get_wer(req.ref_text, hyp_text, mode="cer")
        result_dict.update({
            'ref_txt': wer_res['ref'],
            'hyp_txt': wer_res['hyp'],
            'wer': wer_res['wer'],
            'cer': cer_res['wer'],
        })
    except Exception as e:
        print(f"[ASR Error] {e}")
        result_dict.update({'wer': 1.0, 'cer': 1.0})

    # === 4. F0 Related Metrics (FFE, GPE, VDE) ===
    try:
        f0_model = models['f0']
        metrics = f0_model.compute(req.ref_wav, req.gen_wav)
        result_dict['ffe'] = metrics['ffe']
        result_dict['gpe'] = metrics['gpe']
        result_dict['vde'] = metrics['vde']
    except Exception as e:
        print(f"[F0 Error] {e}")
        result_dict['ffe'] = 0.0
        result_dict['gpe'] = 0.0
        result_dict['vde'] = 0.0

    # === 5. MOS (SingMOS and Sheet) ===
    try:
        mos_model = models['mos']
        mos_res = mos_model.compute_all(req.gen_wav)
        result_dict.update(mos_res)
    except Exception as e:
        print(f"[MOS Error] {e}")
        result_dict.update({
            'singmos': 0.0, 'sheet': 0.0,
        })

    return result_dict

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)