import torch
import torch.nn.functional as F

from src.core.config import config

class Router:

    def __init__(self):
        return

    def select(self, query_embeddings: torch.Tensor, key_embeddings: torch.Tensor) -> torch.Tensor:
        """ Compare the Query embedding with the sequence embeddings and return a mask indicating which key frames to select.

        Args:
            query_embedding (S, D): The embedding of the query.
            key_embeddings (num_key_frames, img_tokens, D): The embeddings of the key frames in the sequence.
        Return:
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

        else:
            raise NotImplementedError(f"Router strategy {config.router_strategy} not implemented.")

        # -------------------------------------------------------
        # Generate Mask
        # -------------------------------------------------------
        _, topk_indices = torch.topk(similarities, k)            
        mask = torch.zeros(N, dtype=torch.bool, device=key_embeddings.device)
        mask[topk_indices] = True
        
        return mask
            