from typing import Dict, Optional, Mapping
from collections import defaultdict

import numpy as np
import torch


def compute_mcts_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    epsilon: float = 1e-6,
    mcts_config: Optional[Mapping] = None,
    index: Optional[np.ndarray] = None,
    norm_adv_by_std_in_grpo: str = True,
    **kwargs
):
    if mcts_config is None:
        mcts_config = {}
    adv_norm_method = mcts_config.get('adv_norm_method', "sibling_mean_std")
    adv_scale_alpha = mcts_config.get('adv_scale_alpha', 1.0)

    # Clone to avoid in-place modification of the original tensor
    scores = token_level_rewards.clone() # (bs, response_length)

    with torch.no_grad():
        bsz = scores.shape[0]
        
        if adv_norm_method == "sibling_mean_std":
            # Sibling normalization: use final token scores for statistics, but apply to all token scores
            if index is None:
                raise ValueError("index is required for sibling_mean_std normalization")
            
            # Step 1: Get the final valid token score for each sequence
            valid_lengths = response_mask.sum(dim=-1).long()  # (bs,)
            final_scores = torch.zeros(bsz, device=scores.device, dtype=scores.dtype)
            
            for i in range(bsz):
                if valid_lengths[i] > 0:
                    # Get the score of the last valid token (0-indexed)
                    last_valid_idx = valid_lengths[i] - 1
                    final_scores[i] = scores[i, last_valid_idx]
            
            # Step 2: Build sibling statistics based on final scores
            id2score = defaultdict(list)
            id2mean = {}
            id2std = {}
            
            for i in range(bsz):
                id2score[index[i]].append(final_scores[i])
            
            for idx in id2score:
                if len(id2score[idx]) == 1:
                    # Single sample in group, no normalization
                    id2mean[idx] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
                    id2std[idx] = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
                elif len(id2score[idx]) > 1:
                    vals = torch.stack(id2score[idx])
                    id2mean[idx] = vals.mean()
                    id2std[idx] = vals.std(unbiased=False)
                else:
                    raise ValueError(f"No score in prompt index: {idx}")
            
            # Step 3: Apply normalization to ALL token scores using final score statistics
            for i in range(bsz):
                # Apply the sibling statistics to normalize each token's original score
                if norm_adv_by_std_in_grpo:
                    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
                else:
                    scores[i] = scores[i] - id2mean[index[i]]
                scores[i] = scores[i] * response_mask[i]
                
        else:
            # Original normalization methods (per-sequence)
            for i in range(bsz):
                if adv_norm_method == "mean_std":
                    valid_score_i = scores[i][response_mask[i] == 1]
                    mean_i = valid_score_i.mean()
                    std_i = valid_score_i.std()
                    scores[i] = (scores[i] - mean_i) / (std_i + epsilon)
                elif adv_norm_method == "alpha_scale":
                    scores[i] = scores[i] * adv_scale_alpha
                elif adv_norm_method == "none":
                    pass  # Keep original scores
                else:
                    raise ValueError(f"Unknown adv_norm_method: {adv_norm_method}. Must be 'mean_std', 'alpha_scale', 'sibling_mean_std', or 'none'")

            scores = scores * response_mask

    return scores, scores
