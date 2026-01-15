from typing import List

import torch

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
            
            seq_emb_list = []
            
            for detail, mrg, msk in zip(detail_list, merged_list, mask):
                if msk:
                    seq_emb_list.append(detail)
                else:
                    seq_emb_list.append(mrg)
            
            seq_emb = torch.cat(seq_emb_list, dim=0)  # (num_frames, img_tokens, D)
            buffer_emb = self.buffer.flush_detail()  # (B, img_tokens, D)
            
            # Concatenate along the temporal/frame dimension (dim 0)
            return torch.cat([seq_emb, buffer_emb], dim=0)