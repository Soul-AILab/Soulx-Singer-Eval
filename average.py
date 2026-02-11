import json
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Compute average evaluation metrics.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--result_file", type=str, required=True)
    return parser.parse_args()

def main():
    args = get_args()

    # Metrics to aggregate
    metrics = {
        # ASR
        'wer': [], 'cer': [],
        # Spectrum
        'mcd': [],
        # Speaker Similarity
        'prompt_gen_cos_sim': [],
        # FFE, GPE, VDE
        'ffe': [], 'gpe': [], 'vde': [],
        # MOS
        'singmos': [], 'sheet': [],
    }

    print(f"Reading from {args.input_file} ...")
    
    valid_count = 0
    has_prompt_language = False
    try:
        with open(args.input_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                if data.get('eval_status') != 'success':
                    continue
                
                valid_count += 1
                if 'prompt_language' in data:
                    has_prompt_language = True
                for key in metrics.keys():
                    if key in data:
                        try:
                            val = float(data[key])
                            metrics[key].append(val)
                        except (ValueError, TypeError):
                            pass
    except FileNotFoundError:
        print(f"Error: File {args.input_file} not found.")
        return

    def _init_metrics():
        return {k: [] for k in metrics.keys()}

    def _avg_metrics(metric_dict):
        out = {}
        for key, val_list in metric_dict.items():
            if len(val_list) > 0:
                avg_val = sum(val_list) / len(val_list)
                out[key] = f"{avg_val:.4f}"
            else:
                out[key] = "0.0000"
        return out

    if has_prompt_language:
        parallel_metrics = _init_metrics()
        cross_metrics = _init_metrics()
        parallel_count = 0
        cross_count = 0

        if valid_count > 0:
            with open(args.input_file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get('eval_status') != 'success':
                        continue

                    if 'prompt_language' not in data:
                        continue

                    src_lang = data.get('language')
                    prompt_lang = data.get('prompt_language')
                    target = parallel_metrics if src_lang == prompt_lang else cross_metrics
                    if src_lang == prompt_lang:
                        parallel_count += 1
                    else:
                        cross_count += 1

                    for key in metrics.keys():
                        if key in data:
                            try:
                                val = float(data[key])
                                target[key].append(val)
                            except (ValueError, TypeError):
                                pass

        out_parallel = _avg_metrics(parallel_metrics)
        out_cross = _avg_metrics(cross_metrics)

        out_dict = {
            "parallel": {
                "infer_num": parallel_count,
                **out_parallel,
            },
            "cross": {
                "infer_num": cross_count,
                **out_cross,
            },
        }

        print(f"\n=== Evaluation Summary (Samples: {valid_count}) ===")
        print(f"parallel infer_num: {parallel_count}")
        print(f"cross infer_num: {cross_count}")
    else:
        out_all = _avg_metrics(metrics)
        out_dict = {
            "infer_num": valid_count,
            **out_all,
        }

        print(f"\n=== Evaluation Summary (Samples: {valid_count}) ===")
        print(f"infer_num: {valid_count}")

    with open(args.result_file, 'w', encoding='utf-8') as fout:
        json.dump(out_dict, fout, indent=4, ensure_ascii=False)

    print(f'\nDetailed averaged results saved to {args.result_file}')

if __name__ == "__main__":
    main()