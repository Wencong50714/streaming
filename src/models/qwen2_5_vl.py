import time

import cv2
import numpy as np
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)

from src.core.config import config
from src.core.metrics import metrics_manager
from src.memory.frames import Frame
from src.models.inference import Model

MAX_NEW_TOKENS = 2048

class Qwen2_5VL_7B(Model):

    def __init__(self, model_path):
        super().__init__(model_path)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="auto", 
            trust_remote_code=True
        )

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        # self.tokenizer.chat_template = (
        #     "{% for message in messages %}"
        #     "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        #     "{% endfor %}"
        #     "{% if add_generation_prompt %}"
        #     "{{ '<|im_start|>assistant\n' }}"
        #     "{% endif %}"
        # )
        self.cached_grids = []

    def _frames_encode(self, frames: np.ndarray) -> torch.Tensor:
        """Encode frames into embedding
        
        Args:
            frames: (num_frames, H, W, C)
            
        Returns:
            embeddings: (num_frames, img_tokens_per_frame, embed_dim)
            image_grid_thw: (num_frames, 3)
        """
        t0 = time.perf_counter()

        num_frames = frames.shape[0]
        image_list = [frames[i] for i in range(frames.shape[0])]

        inputs = self.processor.image_processor(images=image_list, return_tensors="pt")
        
        pixel_values = inputs["pixel_values"].to(self.model.device)
        image_grid_thw = inputs["image_grid_thw"].to(self.model.device)  # (batch, 3) -> (T_chunks, 3)

        with torch.no_grad():
            embeddings = self.model.visual(
                pixel_values, grid_thw=image_grid_thw
            )

        embeddings = embeddings.view(num_frames, -1, embeddings.shape[-1])
        print("Encode {} frames time: {:.2f}s".format(num_frames, time.perf_counter() - t0))
        return embeddings, image_grid_thw
    
    def _video_preprocess(self, video_file_name):
        self.cached_grids = []
        t0 = time.perf_counter()

        cap = cv2.VideoCapture(video_file_name)

        # Compute video original time
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_origin_time = frame_count / fps

        sample_fps = config.sample_fps
        frame_interval = max(1, int(round(fps / sample_fps))) if sample_fps > 0 else 1

        clip_len = config.clip_len

        frames_buffer = []
        sampled_frame_idx = 0
        original_frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if original_frame_idx % frame_interval == 0:
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frames_buffer.append(frame)
                sampled_frame_idx += 1

                if len(frames_buffer) == clip_len:
                    frames_batch = np.stack(frames_buffer)  # (Batch, H, W, C)
                    embeddings, grid = self._frames_encode(frames_batch)
                    embeddings = embeddings
                    self.cached_grids.append(grid)

                    start_id = sampled_frame_idx - len(frames_buffer)

                    for i, frame_emb in enumerate(embeddings):
                        frame_id = start_id + i
                        self.memory.add_frame(Frame(frame_id=frame_id, embedding=frame_emb))
                        print(f"Processed batch frame, ID: {frame_id}")

                    frames_buffer = []

            original_frame_idx += 1

        #  Residual frames
        if len(frames_buffer) > 0:
            frames_batch = np.stack(frames_buffer)
            embeddings, grid = self._frames_encode(frames_batch)
            self.cached_grids.append(grid)

            start_id = sampled_frame_idx - len(frames_buffer)

            for i, frame_emb in enumerate(embeddings):
                frame_id = start_id + i
                self.memory.add_frame(Frame(frame_id=frame_id, embedding=frame_emb))

        cap.release()

        duration = time.perf_counter() - t0
        print("Video Origin Time: {:.2f}s, Sampled Frame Count: {}, Sample FPS: {}, Preprocess Time: {:.2f}s".format(
            video_origin_time, sampled_frame_idx, sample_fps, duration))
        

    def inference(self, video_file_name, prompt) -> str:
        self._video_preprocess(video_file_name)

        device = self.model.device
        key_embeddings = self.memory.get_key_embeddings()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        query_embeddings = self.model.get_input_embeddings()(input_ids).squeeze(0)

        # Use router to generate mask
        mask = self.router.select(query_embeddings, key_embeddings)
        seq_mm_embeddings = self.memory.construct_seq(mask) # (num_frames, img_tokens, D)

        num_video_frames = seq_mm_embeddings.shape[0]
        seq_mm_embeddings = seq_mm_embeddings.flatten(0, 1) # (num_frames * img_tokens, D)
        num_video_tokens = seq_mm_embeddings.shape[0]

        print("num_video_frames:", num_video_frames, "num_video_tokens:", num_video_tokens)

        # Compute video_grid_thw
        all_grids = torch.cat(self.cached_grids, dim=0)
        video_grid_thw = all_grids[:num_video_frames]
        
        # =======================================================
        # Qwen_VL Forward
        # =======================================================
        
        # 1. Get spetial token ids
        vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        video_pad_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        
        # 2. Construct Input token ids
        video_token_ids = [vision_start_id] + [video_pad_id] * num_video_tokens + [vision_end_id]
        video_ids_tensor = torch.tensor(video_token_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        text_inputs = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        input_ids = torch.cat([video_ids_tensor, text_inputs], dim=1)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        video_mask = (input_ids == video_pad_id)
        
        video_embeds = seq_mm_embeddings.to(dtype=inputs_embeds.dtype, device=device)
        inputs_embeds[video_mask] = video_embeds
        
        # 4. Inference
        t0 = time.perf_counter()
        attention_mask = torch.ones_like(input_ids, device=device)
        output_ids = self.model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            image_grid_thw=video_grid_thw, # (T, H, W)
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True
        )
        
        infer_time = time.perf_counter() - t0
        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = self.processor.decode(new_tokens, skip_special_tokens=True)
        metrics_manager.record("Inference Time", infer_time)
        return response
