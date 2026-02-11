import json
import argparse
import os
import requests
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Client for SVS evaluation.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to summary.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save result_zh.json and result_en.json")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000/evaluate")
    return parser.parse_args()

def main():
    args = get_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    file_zh_path = os.path.join(args.output_dir, "result_zh.json")
    file_en_path = os.path.join(args.output_dir, "result_en.json")

    print(f"Reading samples from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    
    print(f"Sending {len(lines)} samples to Evaluation Server ({args.server_url})...")
    
    f_zh = open(file_zh_path, 'w', encoding='utf-8')
    f_en = open(file_en_path, 'w', encoding='utf-8')

    count_zh = 0
    count_en = 0

    for line in tqdm(lines, desc="Evaluating"):
        try:
            line_dict = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        if not line_dict.get('ref_fn') or not line_dict.get('gen_fn'):
            continue

        raw_lang = line_dict.get('language', 'Chinese') 
        
        if raw_lang == 'Chinese':
            target_lang = "zh"
            target_file = f_zh
        else:
            target_lang = "en"
            target_file = f_en

        payload = {
            "ref_text": line_dict['txt'],
            "ref_wav": line_dict['ref_fn'],
            "gen_wav": line_dict['gen_fn'],
            "prompt_wav": line_dict.get('prompt_fn'),
            "lang": target_lang,
        }
        
        final_record = line_dict.copy()
        
        try:
            response = requests.post(args.server_url, json=payload)
            if response.status_code == 200:
                res_data = response.json()
                final_record.update(res_data)
                final_record['eval_status'] = 'success'
            else:
                final_record['eval_status'] = 'fail'
                final_record['error_msg'] = f"Status {response.status_code}: {response.text}"
        except Exception as e:
            final_record['eval_status'] = 'connection_error'
            final_record['error_msg'] = str(e)
            
        target_file.write(json.dumps(final_record, ensure_ascii=False) + '\n')
        target_file.flush()

        if target_lang == 'zh':
            count_zh += 1
        else:
            count_en += 1

    f_zh.close()
    f_en.close()

    print(f"\nEvaluation Done.")
    print(f"  - ZH results: {count_zh} items -> {file_zh_path}")
    print(f"  - EN results: {count_en} items -> {file_en_path}")

if __name__ == "__main__":
    main()