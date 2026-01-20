from typing import List, Union, Optional, Tuple
from dataclasses import dataclass

from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class Frame:
    frame_id: int
    image: Optional[Union[Image.Image, np.ndarray]] = None  # For Visualization, may delete later
    embedding: Optional[torch.Tensor] = None


class KeyFrame:

    def __init__(self, id: int = 0):
        self.id = id
        self.frame_ids: List[int] = []
        self.frame_embeddings: List[torch.Tensor] = []
        self.merged_embedding = None

    def compare_similarity(self, frame_emb: torch.Tensor) -> float:
        assert self.merged_embedding is not None, "Cluster is empty, cannot compare."

        if frame_emb.device != self.merged_embedding.device:
            frame_emb = frame_emb.to(self.merged_embedding.device)

        similarity = F.cosine_similarity(self.merged_embedding, frame_emb, dim=-1)
        # Handle multi-dimensional case by taking mean
        if similarity.dim() > 0:
            similarity = similarity.mean()

        return similarity.item()

    def add_frame(self, frame: Frame):
        new_emb = frame.embedding.detach()
        
        # Update merged embedding
        if self.merged_embedding is None:
            self.merged_embedding = new_emb
        else:
            n = len(self.frame_ids)
            self.merged_embedding = (self.merged_embedding * n + new_emb) / (n + 1)

        self.frame_ids.append(frame.frame_id)
        self.frame_embeddings.append(frame.embedding)

    def get_merged_embedding(self) -> torch.Tensor:
        if self.merged_embedding is None:
            raise ValueError("KeyFrame is empty, no merged embedding available.")
        return self.merged_embedding

    def get_details(self) -> Tuple[torch.Tensor, List[int]]:
        """Concat all frames embeddings into a single tensor
        
        Rertun:
            embeddings (num_frames, img_tokens, D): A tensor containing all frame embeddings.
            frame_ids (List[int]): A list of frame IDs corresponding to the embeddings.
        """
        return torch.stack(self.frame_embeddings, dim=0), self.frame_ids
    
    def get_frame_count(self) -> int:
        return len(self.frame_ids)
