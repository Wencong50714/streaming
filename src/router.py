from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from src.core.config import config

class Router:
    """Router for selecting key frames based on query-key relevance.
    """

    def __init__(self, indexer_checkpoint: Optional[str] = None):
        """Initialize the Router.
        
        Args:
            indexer_checkpoint: Path to trained Indexer checkpoint.
                Required if router_strategy is 'indexer'.
        """
        pass

    def select(self, query_embeddings: torch.Tensor, key_embeddings: torch.Tensor) -> torch.Tensor:
        """Compare the Query embedding with the sequence embeddings and return a mask.

        Args:
            query_embeddings (S, D): The embedding of the query.
            key_embeddings (num_key_frames, img_tokens, D): The embeddings of the key frames.
            
        Returns:
            mask (num_key_frames,): A boolean mask indicating which key frames to select.
        """
        N = key_embeddings.shape[0]
        k = min(config.topk, N)

        # -------------------------------------------------------
        # Strategy 1: Coarse (Global Pooling)
        # -------------------------------------------------------
        if config.router_strategy == "cos_sim_coarse":
            query_vec = query_embeddings.mean(dim=0, keepdim=True) 
            key_vecs = key_embeddings.mean(dim=1) 
            similarities = F.cosine_similarity(query_vec, key_vecs, dim=-1)

        # -------------------------------------------------------
        # Strategy 2: Fine-grained (Max-Mean)
        # High accuracy, suitable for capturing specific objects (Token-wise interaction)
        # -------------------------------------------------------
        elif config.router_strategy == "cos_sim_fine":
            q_norm = F.normalize(query_embeddings, p=2, dim=-1)
            k_norm = F.normalize(key_embeddings, p=2, dim=-1)
            sim_matrix = torch.einsum('sd, npd -> nsp', q_norm, k_norm)
            max_sim_per_word = sim_matrix.max(dim=-1).values
            similarities = max_sim_per_word.mean(dim=-1)

        # -------------------------------------------------------
        # Strategy 3: Cross Attention
        # Uses cross attention mechanism to compute query-key relevance
        # Q = query_embeddings, K = V = key_embeddings
        # Attention score aggregates cross-attention weights for each key frame
        # -------------------------------------------------------
        elif config.router_strategy == "cross-attn":
            # query_embeddings: (S, D)
            # key_embeddings: (N, img_tokens, D)
            D = query_embeddings.shape[-1]
            
            # Normalize embeddings for stable attention computation
            q_norm = F.normalize(query_embeddings, p=2, dim=-1)  # (S, D)
            k_norm = F.normalize(key_embeddings, p=2, dim=-1)    # (N, img_tokens, D)
            
            # Reshape for batch matrix multiplication
            # Q: (S, D) -> (1, S, D)
            # K: (N, img_tokens, D) -> (N, img_tokens, D)
            q_expanded = q_norm.unsqueeze(0)  # (1, S, D)
            
            # Compute attention scores for each key frame
            # attn_scores: (N, S, img_tokens)
            attn_scores = torch.matmul(
                q_expanded.expand(N, -1, -1),  # (N, S, D)
                k_norm.transpose(1, 2)          # (N, D, img_tokens)
            ) / (D ** 0.5)  # Scale by sqrt(d)
            
            # Apply softmax over key tokens dimension
            attn_weights = F.softmax(attn_scores, dim=-1)  # (N, S, img_tokens)
            
            # Aggregate attention weights to get frame-level relevance
            # Average over query tokens and sum over key tokens
            frame_relevance = attn_weights.mean(dim=1).sum(dim=-1)  # (N,)
            
            similarities = frame_relevance

        else:
            raise NotImplementedError(f"Router strategy {config.router_strategy} not implemented.")

        # -------------------------------------------------------
        # Generate Mask (for similarity-based strategies)
        # -------------------------------------------------------
        _, topk_indices = torch.topk(similarities, k)            
        mask = torch.zeros(N, dtype=torch.bool, device=key_embeddings.device)
        mask[topk_indices] = True
        
        return mask
            