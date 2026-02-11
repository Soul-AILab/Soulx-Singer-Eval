infer_dir="./examples"

echo "========================================================"
echo "Processing evaluation for directories in $infer_dir"
echo "========================================================"

if [ ! -d "$infer_dir" ]; then
    echo "Warning: Directory not found: $infer_dir"
    continue
fi

input_file="$infer_dir/summary.json"

if [ ! -f "$input_file" ]; then
    echo "Warning: summary.json not found in $infer_dir"
    continue
fi

echo "1. Running evaluation client..."
python eva_client.py \
    --input_file "$input_file" \
    --output_dir "$infer_dir" \
    --server_url "http://localhost:8000/evaluate"

echo "2. Averaging CHINESE results..."
result_zh="$infer_dir/result_zh.json"
merged_zh="$infer_dir/merged_zh.json"

if [ -s "$result_zh" ]; then
    python average.py \
        --input_file "$result_zh" \
        --result_file "$merged_zh"
else
    echo "No Chinese results found (or empty file)."
fi

echo "3. Averaging ENGLISH results..."
result_en="$infer_dir/result_en.json"
merged_en="$infer_dir/merged_en.json"

if [ -s "$result_en" ]; then
    python average.py \
        --input_file "$result_en" \
        --result_file "$merged_en"
else
    echo "No English results found (or empty file)."
fi

echo "Finished processing $infer_dir"
echo ""

echo "All evaluations completed."