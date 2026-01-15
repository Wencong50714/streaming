import argparse

from src.core.config import config
from src.core.metrics import metrics_manager


parser = argparse.ArgumentParser(description="Video-Language Model Inference")

# ===== Original Arguments =====
parser.add_argument("--video_file", type=str, required=True, help="Path to the input video file")
parser.add_argument("--prompt", type=str, required=True, help="Prompt for the model")
parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model")


if __name__ == "__main__":
    # QWen2_5_VL_7B
    args = parser.parse_args()

    if args.model == "Qwen2_5VL_7B":
        from src.models.qwen2_5_vl import Qwen2_5VL_7B

        model = Qwen2_5VL_7B(args.model_path)

        response = model.inference(args.video_file, args.prompt)
    print(response)

    # Print metrics summary
    metrics_manager.print_summary()