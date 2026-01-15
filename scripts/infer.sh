# Answer A
python -m src.main \
    --model "Qwen2_5VL_7B" \
    --model_path "/home/czh/model_zoo/Qwen2.5-VL-7B-Instruct/" \
    --video_file "test/data/real12_chunk1.mp4" \
    --prompt "What color shirt is the man wearing when he holds the white plate with a piece of raw steak? Options: A. Dark gray. B. Green. C. Blue. D. White. Output A, B, C, or D."

# Answer D
python -m src.main \
    --model "Qwen2_5VL_7B" \
    --model_path "/home/czh/model_zoo/Qwen2.5-VL-7B-Instruct/" \
    --video_file "test/data/real12_chunk2.mp4" \
    --prompt "What object is the person holding and gesturing towards right now? Options: A. A frying pan. B. A bowl of radishes. C. A jar of chili flakes. D. A plate with raw meat. Output A, B, C, or D."

# Answer D
python -m src.main \
    --model "Qwen2_5VL_7B" \
    --model_path "/home/czh/model_zoo/Qwen2.5-VL-7B-Instruct/" \
    --video_file "test/data/real12_chunk3.mp4" \
    --prompt "What is the person holding in their hand? Options: A. A spatula. B. A wooden spoon. C. A fork. D. Red tongs. Output A, B, C, or D."