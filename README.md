# Streaming Video-Language Model

A streaming video understanding system with dynamic memory management and semantic routing.

## System Architecture

1. Encoder Layer
    - Encode video in the granularity of clip (default 8 frames)
    - Dynamic Cluster
2. Memory Management
    - Key frame buffer with similarity-based clustering
    - Efficient memory access patterns

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Inference

```bash
python -m src.main \
    --video_file /path/to/video.mp4 \
    --prompt "Describe this video" \
    --model Qwen2_5VL_7B \
    --model_path /path/to/Qwen2.5-VL-7B-Instruct
```

## OVO-Bench Evaluation

This model can be evaluated on OVO-Bench (Online Video-Language Understanding Benchmark).

### Prerequisites

1. Download OVO-Bench data:
   ```bash
   cd bench/OVO-Bench
   # Follow instructions in bench/OVO-Bench/README.md to download data
   ```

2. Prepare chunked videos:
   ```bash
   bash scripts/chunk_video.sh
   ```

### Run Evaluation

**Option 1: Use the convenience script (recommended)**

```bash
# Run full evaluation (inference + scoring)
bash scripts/eval_ovo_bench.sh --model_path /path/to/model

# Run specific tasks only
bash scripts/eval_ovo_bench.sh --model_path /path/to/model --tasks "EPM ASI HLD"

# Run inference only (skip scoring)
bash scripts/eval_ovo_bench.sh --model_path /path/to/model --inference-only

# Run scoring only (requires previous inference results)
bash scripts/eval_ovo_bench.sh --score-only
```

**Option 2: Run manually**

```bash
cd bench/OVO-Bench

# Step 1: Inference
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python inference.py \
    --anno_path data/ovo_bench_new.json \
    --video_dir data/ \
    --chunked_dir data/chunked_videos \
    --result_dir results \
    --mode offline \
    --task STU \
    --model StreamingQwen2_5VL \
    --model_path "Qwen/Qwen2.5-VL-7B-Instruct"

# Step 2: Scoring
python score.py \
    --result_dir results \
    --model StreamingQwen2_5VL \
    --mode offline
```

### Available Tasks

| Category | Tasks | Description |
|----------|-------|-------------|
| Backward | EPM, ASI, HLD | Backward-tracing tasks |
| Realtime | STU, OJR, ATR, ACR, OCR, FPD | Real-time perception tasks |
| Forward | REC, SSR, CRR | Forward prediction tasks |

### Configuration

The model behavior can be configured in `config.yaml`:

```yaml
# Video Sampling
sample_fps: 1      # Video sampling FPS
clip_len: 4        # Video clip length

# Query Embedding Match
topk: 8            # Top K matches for embedding
router_strategy: "cos_sim_fine"  # Router strategy: cos_sim_coarse, cos_sim_fine, indexer, cross-attn

# Memory Parameters
key_frame_buffer_size: 5         # Size of key frame buffer
key_frame_match_ratio: 0.5       # Match ratio for key frames
```

### Router Strategies

The system supports multiple routing strategies for selecting relevant key frames:

- **`cos_sim_coarse`**: Global pooling + cosine similarity. Fast but less accurate.
- **`cos_sim_fine`**: Token-wise max-mean cosine similarity. High accuracy for capturing specific objects.
- **`indexer`**: Learnable indexer with trained W_q, W_k projections. Requires training with `train_indexer.py`.
- **`cross-attn`**: Cross attention mechanism where query tokens attend to key frame tokens. Balances accuracy and computational efficiency.

## Project Structure

```
streaming/
├── src/
│   ├── main.py              # Main entry point
│   ├── router.py            # Query-based frame routing
│   ├── core/
│   │   ├── config.py        # Configuration loader
│   │   └── metrics.py       # Metrics tracking
│   ├── memory/
│   │   ├── buffer.py        # Key frame buffer
│   │   ├── frames.py        # Frame data structures
│   │   └── memory.py        # Memory management
│   └── models/
│       ├── inference.py     # Base model interface
│       └── qwen2_5_vl.py    # Qwen2.5-VL implementation
├── bench/
│   └── OVO-Bench/           # OVO-Bench evaluation framework
├── scripts/
│   ├── eval_ovo_bench.sh    # OVO-Bench evaluation script
│   └── infer.sh             # Inference script
└── config.yaml              # Model configuration