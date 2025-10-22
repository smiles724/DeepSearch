from verl import DataProto


class DeepSearchRewardManager:
    """The reward manager."""

    def __init__(
            self,
            tokenizer = None,
            num_examine = None,
            compute_score=None,
            reward_fn_key="data_source"
    ) -> None:
        # everything is placeholder, just for init compatibility
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto,  return_dict: bool = False):
        # Verify that required fields exist
        if "token_level_rewards" not in data.batch.keys():
            raise KeyError("token_level_rewards not found in data.batch.")

        # Extract pre-computed token-level rewards
        token_level_rewards = data.batch["token_level_rewards"]

        # Verify shape consistency
        if "responses" in data.batch.keys():
            responses = data.batch["responses"]
            assert token_level_rewards.shape == responses.shape, (
                f"Shape mismatch: token_level_rewards {token_level_rewards.shape} "
                f"vs responses {responses.shape}"
            )

        # Optional post-processing can be done here
        # For example: mask out rewards at padding positions
        if "attention_mask" in data.batch.keys():
            # Get the attention_mask for the response portion
            full_attention_mask = data.batch["attention_mask"]
            prompt_length = data.batch["prompts"].shape[1]
            response_mask = full_attention_mask[:, prompt_length:]

            # Set rewards at padding positions to 0
            token_level_rewards = token_level_rewards * response_mask.float()

        if not return_dict:
            return token_level_rewards

        # Compute some statistics for monitoring
        reward_stats = {
            "mean_reward": token_level_rewards.mean().item(),
            "max_reward": token_level_rewards.max().item(),
            "min_reward": token_level_rewards.min().item(),
            "non_zero_ratio": (token_level_rewards != 0).float().mean().item(),
        }

        # Return in dictionary format
        return {
            "reward_tensor": token_level_rewards,
            "reward_extra_info": {
                "reward_source": "mcts_q_values",
                "reward_stats": reward_stats,
            }
        }
