from typing import List
from statistics import mean, median

import torch

from src.core.metrics import metrics_manager
from src.memory.buffer import KeyFrameBuffer, Frame, KeyFrame

class Memory:
    def __init__(self):
        self.buffer = KeyFrameBuffer()
        self.key_frames: List[KeyFrame] = []

    def add_frame(self, frame_item: Frame) -> KeyFrame:
        kf = self.buffer.add_frame(frame_item)
        if kf is not None:
            self.key_frames.append(kf)

    def get_key_embeddings(self) -> torch.Tensor:
        """For Indexer Use. Get all key frame embeddings concatenated into a single tensor.

        Return:
            key_embeddings (num_frame, img_tokens, D): A tensor containing all key frame embeddings.
        """
        # Spetial Case, all key frames are still in buffer
        key_frames = self.buffer.flush_key_frame() if self.key_frames == [] else self.key_frames
        return torch.stack([kf.get_merged_embedding() for kf in key_frames], dim=0)

    def construct_seq(self, mask: torch.Tensor) -> torch.Tensor:
        """Construct a sequence of key frame embeddings based on the mask.

        Args:
            mask (K, ): A boolean mask indicating which key frames to include in the sequence.

        Returns:
            video_seq (TotalFrames, img_tokens, D): A tensor containing the selected key frame embeddings.
        """
        print("Key Frame Num = {}".format(len(self.key_frames)))

        if not self.key_frames:
            buffer_emb = self.buffer.flush_detail()
            return buffer_emb
        else:
            # Prepare 4D tensors for Triton
            detail_list = [kf.get_details()[0] for kf in self.key_frames] # (num_key_frams, var_frames, img_tokens, D)
            merged_list = [kf.get_merged_embedding().unsqueeze(0) for kf in self.key_frames]
            
            assert len(detail_list) == len(merged_list) == mask.shape[0]
            
            # Build adjacency mask: msk=0 positions that are adjacent to msk=1 positions
            k = mask.shape[0]
            adjacent_mask = torch.zeros(k, dtype=torch.bool)
            
            for i in range(k):
                if mask[i]:
                    # Current position is 1, mark itself as selected
                    adjacent_mask[i] = True
                    # Mark adjacent positions (i-1 and i+1) if they are 0
                    if i > 0 and not mask[i-1]:
                        adjacent_mask[i-1] = True
                    if i < k-1 and not mask[i+1]:
                        adjacent_mask[i+1] = True
            
            seq_emb_list = []
            meg_emb_cnt = 0
            
            for i, (detail, mrg, msk) in enumerate(zip(detail_list, merged_list, mask)):
                if not adjacent_mask[i]:
                    # Skip this position if it's not adjacent to any mask=1 position
                    continue
                    
                if msk:
                    seq_emb_list.append(detail)
                else:
                    seq_emb_list.append(mrg)
                    meg_emb_cnt += 1
            
            seq_emb = torch.cat(seq_emb_list, dim=0) if seq_emb_list else torch.empty(0, detail_list[0].shape[1], detail_list[0].shape[2])  # (num_frames, img_tokens, D)
            buffer_emb = self.buffer.flush_detail()  # (B, img_tokens, D)
            
            # Add Statistics
            if self.key_frames:
                frame_counts = [kf.get_frame_count() for kf in self.key_frames]
                avg_frame_count = mean(frame_counts)
                selected_detail_frame_ids = [kf.get_details()[1] for kf, msk in zip(self.key_frames, mask) if msk]

                median_frame_count = median(frame_counts)
                metrics_manager.record("Key Frame Avg Frames Count", avg_frame_count)
                metrics_manager.record("Key Frame Median Frames Count", median_frame_count)
                metrics_manager.record("Buffer Frame Count", buffer_emb.shape[0])
                metrics_manager.record("Merged Embedding Count", meg_emb_cnt)
                metrics_manager.record("Selected Detailed Frame IDs", selected_detail_frame_ids)
                
            # Concatenate along the temporal/frame dimension (dim 0)
            return torch.cat([seq_emb, buffer_emb], dim=0)