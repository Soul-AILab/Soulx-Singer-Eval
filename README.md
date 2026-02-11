<div align="center">
	<h1>🎤 SoulX-Singer-Eval</h1>
	<p>
		Evaluation suite for zero-shot Singing Voice Synthesis (SVS) systems, covering aesthetic appeal, signal quality, pronunciation accuracy, speaker similarity, and melodic precision.
	</p>
	<p>
		<a href="https://huggingface.co/datasets/Soul-AILab/SoulX-Singer-Eval-Dataset"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Eval--Dataset-Hugging%20Face-ffda16" alt="Eval Dataset"></a>
		<a href="https://arxiv.org/abs/2602.07803"><img src="https://img.shields.io/badge/arXiv-2602.07803-b31b1b" alt="SoulX-Singer arXiv"></a>
		<a href="https://github.com/Soul-AILab/SoulX-Singer"><img src="https://img.shields.io/badge/SoulX--Singer-Repository-black" alt="SoulX-Singer Repo"></a>
	</p>
</div>

---

## 📊 Metrics Overview

### 1. Singing Aesthetics

We incorporate two MOS (Mean Opinion Score) prediction models to evaluate the subjective appeal of synthesized singing.

* **SingMOS-Pro**: A specialized MOS predictor for singing voice, focusing on professional vocal attributes.
* **Sheet-SSQA**: Simple Hierarchy-aware Enhancement-based Tool for Speech Subjective Quality Assessment.

### 2. Spectral Quality

* **Mel Cepstral Distortion (MCD)**: Measures the spectral distance between predicted and ground-truth audio.



### 3. Pronunciation Intelligibility

* **WER/CER**: Evaluates accuracy using ASR models.
* **English**: [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3).
* **Chinese**: [Paraformer-large](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch).
* *Note: Punctuation is removed before computation.*



### 4. Speaker Similarity

* **Speaker-Sim (Cosine Similarity)**: Computes cosine similarity between prompt and generated voices.
* Model: [WavLM-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv). You can pass a local path or model id via `model_path_or_id` when initializing `SVPipeline`.



### 5. Melodic Accuracy

* **FFE / GPE / VDE**: Frame Error, Gross Pitch Error, and Voicing Decision Error.

---

## 🛠 Installation


```bash
conda create -n soulx-singer-eval python=3.10
conda activate soulx-singer-eval
pip install -r requirements.txt
```

## 📦 Model Checkpoints

Before running evaluation, download the following files and place them under the `ckpt/` directory:

* `all7-sslmos-mdf-2337-config.yml`  
	https://github.com/unilight/sheet/releases/download/v0.1.0/all7-sslmos-mdf-2337-config.yml
* `all7-sslmos-mdf-2337-checkpoint-86000steps.pkl`  
	https://github.com/unilight/sheet/releases/download/v0.1.0/all7-sslmos-mdf-2337-checkpoint-86000steps.pkl
* `ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth`  
	https://github.com/South-Twilight/SingMOS/releases/download/ckpt_v3/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth

> Note: If HuggingFace is unreachable in your environment, `s3prl` may fail to download SSL checkpoints because its URLs are hard-coded. You can patch `s3prl` to use the `hf-mirror` domain by replacing `https://huggingface.co/` with `https://hf-mirror.com/` in the s3prl source (`s3prl/upstream/wav2vec2/hubconf.py`).

## 📚 Datasets

Due to the absence of a widely adopted SVS benchmark, we provide two complementary evaluation datasets to assess open-source and zero-shot conditions: **GMO-SVS** and **SoulX-Singer-Eval**.

HuggingFace: https://huggingface.co/datasets/Soul-AILab/SoulX-Singer-Eval-Dataset

### GMO-SVS

GMO-SVS is built upon three public SVS corpora: **G**TSinger, **M**4Singer, and **O**pencpop. For M4Singer and Opencpop, we adopt their official test splits. GTSinger contributes English and Mandarin songs from multiple singers with diverse techniques and styles. In total, GMO-SVS contains **802 samples**.

For each song, the first sentence is used as the acoustic prompt, and the remaining content is synthesized by evaluated models. Ground-truth recordings of the prompt singers are preserved to evaluate pronunciation accuracy, prosodic consistency, and overall synthesis quality. None of these open-source datasets are used in SoulX-Singer training, ensuring fair evaluation.

### SoulX-Singer-Eval

SoulX-Singer-Eval is a newly collected dataset for zero-shot generalization on unseen speakers. It contains **100 singing segments** from **50 distinct individuals** (25 Mandarin and 25 English speakers), with **2 segments per speaker**. Mandarin data are collected from recruited professional and amateur singers who consented to open-source their voice data for academic purposes. English segments are sliced and filtered from the multitrack *Mixing Secrets* dataset. All segments are manually annotated with precise melody to meet prompt requirements for zero-shot SVS models.

Target lyrics and melodies for synthesis are randomly selected from 15 Mandarin and 15 English tracks in GMO-SVS. This introduces speakers unseen by baseline models and provides a rigorous benchmark for timbre cloning and style transfer.


## 🚀 Usage

### 1) Prepare your samples

Follow the structure in [examples/summary.json](examples/summary.json). Each line is a JSON record with:

* `txt`: reference transcript
* `ref_fn`: reference wav path
* `gen_fn`: generated wav path
* `prompt_fn`: prompt wav path
* `language`: Chinese or English
* `prompt_language`: language of the prompt

### 2) Start the evaluation server

```bash
bash eva_server_run.sh
```

### 3) Run evaluation (recommended script)

Edit [eva_client_run.sh](eva_client_run.sh) and set `infer_dir` to the folder that contains a `summary.json` file, then run:

```bash
bash eva_client_run.sh
```

The script will generate:

* `result_zh.json` / `result_en.json`
* `merged_zh.json` / `merged_en.json`

### 4) Run evaluation (manual)

```bash
python eva_client.py --input_file examples/summary.json --output_dir examples
```

Results will be written to:

* `examples/result_zh.json`
* `examples/result_en.json`

Then aggregate:

```bash
python average.py --input_file examples/result_zh.json --result_file examples/merged_zh.json
```

---

## 🔗 Acknowledgements

This project integrates components from the following repositories:

* [TTS-Evaluation](https://github.com/Shengqiang-Li/TTS-Evaluation)
* [TTS-Objective-Metric](https://github.com/AI-Unicamp/TTS-Objective-Metrics)
* [SingMOS](https://github.com/South-Twilight/SingMOS)
* [Sheet-SSQA](https://github.com/unilight/sheet)

