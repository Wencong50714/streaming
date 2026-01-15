import torch

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

        if config.router_strategy == "cos_sim":
            
            query_vec = query_embeddings.mean(dim=0, keepdim=True) 
            key_vecs = key_embeddings.mean(dim=1) 
            similarities = torch.nn.functional.cosine_similarity(query_vec, key_vecs, dim=-1)

            N = key_embeddings.shape[0]
            k = min(config.topk, N)

            _, topk_indices = torch.topk(similarities, k)            
            mask = torch.zeros(N, dtype=torch.bool, device=key_embeddings.device)
            mask[topk_indices] = True
            
            return mask
        else:
            raise NotImplementedError(f"Router strategy {config.router_strategy} not implemented.")
            