from typing import List

import torch

from src.core.config import config
from src.memory.frames import Frame, KeyFrame
from src.core.metrics import metrics_manager


class KeyFrameBuffer:

    def __init__(self, max_size: int = None):
        if max_size is None:
            max_size = config.key_frame_buffer_size
        self.max_size = max_size
        self.buffer: List[KeyFrame] = []
        self.cur_id = 0

    def _new_key_frame(self, frame: Frame) -> None:
        """Create a new KeyFrame with a unique ID."""
        new_kf = KeyFrame(id=self.cur_id)
        self.cur_id += 1
        new_kf.add_frame(frame)
        self.buffer.append(new_kf)

    def add_frame(self, frame: Frame) -> KeyFrame:
        """Add a new frame to the buffer.

        Args:
            frame: The frame item containing frame_id, image, and embedding.

        Returns:
            The evicted KeyFrame if buffer was at max_size and a new KeyFrame was added, otherwise None.
        """

        if len(self.buffer) == 0:
            self._new_key_frame(frame)
        else:
            similarities = [kf.compare_similarity(frame.embedding) for kf in self.buffer]
            max_similarity = max(similarities)

            if max_similarity < config.key_frame_match_ratio:
                self._new_key_frame(frame)
                if len(self.buffer) > self.max_size:
                    evicted_kf = self.buffer.pop(0)
                    return evicted_kf  # other case return None
            else:
                best_kf_index = similarities.index(max_similarity)
                best_kf = self.buffer[best_kf_index]
                
                # Check if the key frame has reached max_frame_set_num limit
                if config.max_frame_set_num is not None and best_kf.get_frame_count() >= config.max_frame_set_num:
                    # Cannot merge, create a new key frame instead
                    self._new_key_frame(frame)
                    if len(self.buffer) > self.max_size:
                        evicted_kf = self.buffer.pop(0)
                        return evicted_kf
                else:
                    best_kf.add_frame(frame)
                    # Record merge metrics
                    metrics_manager.record("Merged Frame IDs", frame.frame_id)
                    metrics_manager.record("Merge Similarity", max_similarity)
                    metrics_manager.record("Total Merge Count")

    def flush_key_frame(self) -> List[KeyFrame]:
        return self.buffer

    def flush_detail(self) -> torch.Tensor:
        """Convert all key frames to detailed embeddings and clear the buffer. embeddings should sort by frame_id
        
        Returns:
            embeddings: (num_frames, img_tokens, D): A tensor containing all detailed embeddings.
        """

        if not self.buffer:
            return torch.tensor([])

        details = [kf.get_details() for kf in self.buffer]
        details.sort(key=lambda x: x[1])
        sorted_embeddings = [x[0] for x in details]

        self.buffer = []
        return torch.cat(sorted_embeddings, dim=0)  # (num_frames, img_tokens, D)
