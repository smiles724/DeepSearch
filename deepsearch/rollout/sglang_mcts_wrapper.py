import json
import math
from copy import deepcopy
import time

import numpy as np
import torch
import logging
import os
from typing import Dict
from omegaconf import DictConfig, OmegaConf

from deepsearch.utils.data_proto import slice_from_data_proto, stack_data_protos, \
    deduplicate_prompts, pad_list_data_protos, mask_select_from_data_proto
from deepsearch.utils.mcts import MCTSAgent, get_values_from_node

from verl import DataProto
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class DeepSearchSGLangRollout:

    def __init__(
            self,
            sglang_rollout_cls,
            sglang_rollout_kwargs: Dict,
            tokenizer,
            mcts_config: DictConfig,
    ):
        sglang_rollout_kwargs = deepcopy(sglang_rollout_kwargs)
        self.mcts_config = mcts_config

        # mcts algorithm part
        self.lambda_puct = self.mcts_config.get('lambda_puct', 2)

        self.backup_threshold = self.mcts_config.get('backup_threshold', 0.1)
        self.q_correct = self.mcts_config.get('q_correct', 0.1)
        self.max_depth = self.mcts_config.get('max_depth', 32)
        self.terminate_num = sglang_rollout_kwargs["config"].n
        self.expansion_width = self.mcts_config.get('expansion_width', self.terminate_num)
        self.incomplete_score = self.mcts_config.get('incomplete_score', -1.0)
        self.dynamic_width_config = self.mcts_config.get('dynamic_width_config', None)

        # only valid for self.backtrace_method='global'
        self.global_lambda1 = self.mcts_config.get('global_lambda1', 0.4)
        self.global_lambda2 = self.mcts_config.get('global_lambda2', 0.4)
        self.global_lambda3 = self.mcts_config.get('global_lambda3', 0.01)
        self.global_lambda4 = self.mcts_config.get('global_lambda4', 0.0)
        self.depth_bonus_type = self.mcts_config.get('depth_bonus_type', "sqrt")
        self.involve_uncertainty_bonus = self.mcts_config.get('involve_uncertainty_bonus', False)
        self.force_to_frontier = self.mcts_config.get('force_to_frontier', True)
        self.use_step_wise_score = self.mcts_config.get('use_step_wise_score', True)

        overlong_buffer_cfg = self.mcts_config.get('overlong_buffer_cfg', {})
        # Create a new regular dict with defaults
        self.overlong_buffer_cfg = OmegaConf.create({
            "enable": overlong_buffer_cfg.get("enable", False),
            "len": overlong_buffer_cfg.get("len", 4096),
            "penalty_factor": overlong_buffer_cfg.get("penalty_factor", 1.0),
            "log": overlong_buffer_cfg.get("log", False)
        })
        if self.overlong_buffer_cfg.enable:
            if not self.use_step_wise_score:
                logger.warning("If step-wise scoring is disabled, overlong_buffer_cfg in mcts_config won't be effective! Please register it outside mcts_config.")
            else:
                logger.info(f"overlong buffer activated for MCTS step-wise scoring: {self.overlong_buffer_cfg}")

        self.adv_temperature = self.mcts_config.get("adv_temperature", 1.0)  # default to disable the range constraints
        self.q_value_max = self.mcts_config.get("q_value_max", 1.0)
        if self.adv_temperature > 0:
            print(f"Activate Tanh range constraints by adv_temperature={self.adv_temperature} and q_value_max={self.q_value_max}")

        self.prompt_length = sglang_rollout_kwargs["config"].prompt_length
        self.response_length = sglang_rollout_kwargs["config"].response_length
        self.step_response_length = self.response_length // self.max_depth

        self.backtrace_method = self.mcts_config.get('backtrace_method', "global")
        assert self.backtrace_method in ["uct", "global"], self.backtrace_method
        self.deterministic_sample = self.mcts_config.get('deterministic_sample', True)

        self.reward_fn = get_custom_reward_fn(mcts_config)
        if self.use_step_wise_score:
            assert self.reward_fn is not None, "`custom_reward_function` must be given in mcts_config if `use_step_wise_score` is set!"

        self.replay_buffer = mcts_config.get("replay_buffer", None)
        if self.replay_buffer is not None:
            with open(self.replay_buffer, "r") as f:
                self.replay_buffer = json.load(f)

        assert sglang_rollout_kwargs["config"].response_length % self.max_depth == 0
        sglang_rollout_kwargs["config"].response_length = self.step_response_length
        sglang_rollout_kwargs["config"].n = 1

        self.sglang_rollout = sglang_rollout_cls(**sglang_rollout_kwargs)
        self.tokenizer = tokenizer
        self.batch_size = self.mcts_config.get('batch_size', -1)

        self.tp_size = self.sglang_rollout._tp_size
        self.tp_rank = self.sglang_rollout._tp_rank
        self.global_rank = self.sglang_rollout._rank
        self.global_size = torch.distributed.get_world_size()
        self.tp_mesh = self.sglang_rollout._device_mesh_cpu["tp"]
        
        # Ensure global_size is divisible by tp_size
        assert self.global_size % self.tp_size == 0, \
            f"global_size ({self.global_size}) must be divisible by tp_size ({self.tp_size})"

        self.config = self.sglang_rollout.config
        self.pad_token_id = self.sglang_rollout.pad_token_id
        self._engine = self.sglang_rollout._engine

        self.visualize_debug = self.mcts_config.get('visualize_debug', False)
        
        logger.info(
            f"[RANK {self.global_rank}-{self.global_size} TP-RANK {self.tp_rank}-{self.tp_size}] "
            f"Initialized MCTSvLLMRollout for with config: {self.mcts_config}"
        )

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        unique_prompts = deduplicate_prompts(prompts)
        # when input prompts have already been repeated
        if len(prompts.batch) != len(unique_prompts.batch):
            assert len(unique_prompts.batch) * self.terminate_num == len(prompts.batch), (
                len(unique_prompts.batch), len(prompts.batch), self.expansion_width
            )

        trajectories = self.get_trajectories_from_prompts(input_prompts=unique_prompts, **kwargs)
        return trajectories

    def get_trajectories_from_prompts(self, input_prompts: DataProto, **kwargs) -> DataProto:
        batch_size = self.batch_size
        if batch_size == -1:
            batch_size = len(input_prompts.batch)

        in_buffer_outputs = None
        registered_in_buffer = None
        if self.replay_buffer is not None:
            prompt_ids = input_prompts.non_tensor_batch.get('index')
            registered_in_buffer = np.array([p_id in self.replay_buffer for p_id in prompt_ids], dtype=bool)

            # split the input_prompts into two parts:
            # 1. index founded in replay buffer; (directly rollout & replace)
            # 2. index not founded in replay buffer (MCTS rollout)
            # after two branch, ensemble them by the original index order

            in_buffer_prompts = mask_select_from_data_proto(data_proto=input_prompts, valid_mask=registered_in_buffer)
            out_buffer_prompts = mask_select_from_data_proto(data_proto=input_prompts, valid_mask=~registered_in_buffer)

            logger.info(
                f"[RANK {self.global_rank} TP-RANK {self.tp_rank}-{self.tp_size}] "
                f"in-buffer / out-buffer = {len(in_buffer_prompts.batch)}/{len(out_buffer_prompts.batch)}"
            )

            # extract the correct response str in the replay buffer by index[i]
            # turn the response str into responses, input_ids, attention_mask, position_ids, rm_scores by self.tokenizer
            # 1. pick up all indices registered in the replay buffer
            # 2. for the registered ids, directly call self.sglang_rollout.generate_sequences() by self.response_length
            # 3. replace the incorrect responses by the registered responses in the buffer

            # IMPORTANT: Always call generate_sequences to ensure all ranks participate in collective ops
            # even when in_buffer_prompts is empty, to avoid NCCL timeout
            if len(in_buffer_prompts.batch) > 0:
                old_response_length = self.sglang_rollout.config.response_length
                self.sglang_rollout.config.response_length = self.response_length
                in_buffer_outputs = self.sglang_rollout.generate_sequences(
                    in_buffer_prompts, max_new_tokens=self.response_length, n=self.terminate_num, **kwargs
                )
                self.sglang_rollout.config.response_length = old_response_length

                in_buffer_ids = in_buffer_prompts.non_tensor_batch.get("index")
                for prompt_idx, prompt_id in enumerate(in_buffer_ids):
                    correct_candidates = self.replay_buffer[prompt_id][:self.terminate_num]
                    correct_cand_inputs = self.tokenizer(
                        correct_candidates,
                        add_special_tokens=False,
                        return_tensors="pt",
                        padding='max_length',  # ← Changed: use max_length padding
                        truncation=True,
                        max_length=self.response_length,
                    )

                    cand_start = prompt_idx * self.terminate_num
                    cand_end = cand_start + len(correct_candidates)

                    correct_cand_input_ids = correct_cand_inputs.input_ids.to(in_buffer_outputs.batch["responses"].device)
                    correct_cand_attention_mask = correct_cand_inputs.attention_mask.to(in_buffer_outputs.batch["responses"].device)

                    in_buffer_outputs.batch["responses"][cand_start:cand_end] = correct_cand_input_ids
                    cand_input_attention_mask = in_buffer_outputs.batch["attention_mask"][:, :self.prompt_length][cand_start:cand_end]
                    in_buffer_outputs.batch["attention_mask"][cand_start:cand_end] = torch.cat(
                        (cand_input_attention_mask, correct_cand_attention_mask), dim=-1
                    )
                    cand_input_prompts = in_buffer_outputs.batch["prompts"][cand_start:cand_end]
                    in_buffer_outputs.batch["input_ids"][cand_start:cand_end] = torch.cat(
                        (cand_input_prompts, correct_cand_input_ids), dim=-1
                    )

                rm_scores = []
                for i in range(len(in_buffer_outputs.batch)):
                    output_i = slice_from_data_proto(data_proto=in_buffer_outputs, index=i)

                    mask = (output_i.batch["input_ids"][0] != self.pad_token_id)
                    left_idx = mask.to(torch.uint8).argmax().item()
                    reversed_mask = mask.flip(dims=[0])
                    right_pad_count = reversed_mask.to(torch.uint8).argmax().item()
                    right_idx = len(mask) - right_pad_count if right_pad_count > 0 or mask[-1] else len(mask)

                    candidate_token_ids = output_i.batch["input_ids"][0][left_idx:right_idx].tolist()
                    candidate_text = self.tokenizer.decode(candidate_token_ids)
                    ground_truth = output_i.non_tensor_batch["reward_model"][0]["ground_truth"]
                    reward_dict = self.reward_fn(solution_str=candidate_text, ground_truth=ground_truth)

                    has_eos = output_i.batch["attention_mask"][0][-1] == 0
                    parsed_ans = reward_dict["pred"]
                    correct = reward_dict["acc"]
                    if (not has_eos) and parsed_ans == "[INVALID]":
                        final_value = self.incomplete_score
                    elif correct:
                        final_value = 1.0
                    else:
                        final_value = -1.0

                    if self.overlong_buffer_cfg.enable:
                        overlong_buffer_len = self.overlong_buffer_cfg.len
                        expected_len = self.response_length - overlong_buffer_len
                        valid_response_length = output_i.batch["attention_mask"][0, self.prompt_length:].sum().item()

                        exceed_len = valid_response_length - expected_len
                        overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                        overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                        final_value += overlong_reward

                    rm_scores.append(final_value)

                rm_scores = torch.tensor(
                    rm_scores, dtype=torch.float32, device=in_buffer_outputs.batch["responses"].device
                ).unsqueeze(1)
                in_buffer_outputs.batch["rm_scores"] = rm_scores.expand_as(in_buffer_outputs.batch["responses"]).clone()
                resp_attention_mask_bool = in_buffer_outputs.batch["attention_mask"][:, self.prompt_length:] == 0
                in_buffer_outputs.batch["rm_scores"][resp_attention_mask_bool] = 0.0

            else:
                # When in_buffer_prompts is empty, still need to call generate_sequences
                # to participate in collective operations (avoid NCCL timeout)
                # Use a dummy prompt from the original input_prompts
                dummy_prompts = slice_from_data_proto(data_proto=input_prompts, index=0)
                old_response_length = self.sglang_rollout.config.response_length
                self.sglang_rollout.config.response_length = self.response_length
                dummy_outputs = self.sglang_rollout.generate_sequences(
                    dummy_prompts, max_new_tokens=self.response_length, n=self.terminate_num, **kwargs
                )
                self.sglang_rollout.config.response_length = old_response_length
                # Discard dummy_outputs as they are not used

        else:
            out_buffer_prompts = input_prompts

        traj_stack = []
        for batch_start in range(0, len(out_buffer_prompts.batch), batch_size):
            batch_end = min(batch_start + batch_size, len(out_buffer_prompts.batch))
            batch_prompts = slice_from_data_proto(
                out_buffer_prompts, index=batch_start, stride=batch_end - batch_start
            )
            batch_traj = self._run_mcts_for_prompt(prompt=batch_prompts, **kwargs)

            traj_stack.append(batch_traj)
            logger.info(
                f"[RANK {self.global_rank} TP-RANK {self.tp_rank}-{self.tp_size}] "
                f"Finished MCTS rollout samples ({batch_end}/{len(out_buffer_prompts.batch)})"
            )

        trajectories = stack_data_protos(traj_stack)

        if in_buffer_outputs is not None:
            traj_stack = []
            in_buffer_cursor = 0
            out_buffer_cursor = 0
            for regi_flag in registered_in_buffer:
                if regi_flag:
                    traj_stack.append(
                        slice_from_data_proto(
                            data_proto=in_buffer_outputs,
                            index=in_buffer_cursor,
                            stride=self.terminate_num,
                        )
                    )
                    in_buffer_cursor += self.terminate_num
                else:
                    traj_stack.append(
                        slice_from_data_proto(
                            data_proto=trajectories,
                            index=out_buffer_cursor,
                            stride=self.terminate_num,
                        )
                    )
                    out_buffer_cursor += self.terminate_num

            trajectories = stack_data_protos(traj_stack)

        return trajectories

    def _run_mcts_for_prompt(self, prompt: DataProto, **kwargs) -> DataProto:
        batch_size = len(prompt.batch)

        # Record MCTS start time
        mcts_start_time = time.time()

        if self.tp_rank == 0:
            logger.info(
                f"[RANK {self.global_rank} TP-RANK {self.tp_rank}-{self.tp_size}] "
                f"Starting MCTS for batch_size={batch_size}"
            )

        agent_list = []
        for prm_idx in range(batch_size):
            agent_prompt = slice_from_data_proto(prompt, index=prm_idx)

            # raw_prompt_ids will be inferred in vllm_rollout.generate_sequences by input_ids
            ori_raw_prompt_ids = agent_prompt.non_tensor_batch.pop("raw_prompt_ids", None)
            agent = MCTSAgent(
                data_proto=agent_prompt,
                pad_token_id=self.pad_token_id,
                tokenizer=self.tokenizer,
                lambda_puct=self.lambda_puct,
                expansion_width=self.expansion_width,
                max_depth=self.max_depth,
                deterministic_sample=self.deterministic_sample,
                ori_raw_prompt_ids=ori_raw_prompt_ids,
                logger=logger,
                backtrace_method=self.backtrace_method,
                backup_threshold=self.backup_threshold,
                q_correct=self.q_correct,
                step_response_length=self.step_response_length,
                terminate_num=self.terminate_num,
                global_lambda1=self.global_lambda1,
                global_lambda2=self.global_lambda2,
                global_lambda3=self.global_lambda3,
                global_lambda4=self.global_lambda4,
                depth_bonus_type=self.depth_bonus_type,
                involve_uncertainty_bonus=self.involve_uncertainty_bonus,
                force_to_frontier=self.force_to_frontier,
                reward_fn=self.reward_fn,
                incomplete_score=self.incomplete_score,
                dynamic_width_config=self.dynamic_width_config,
            )
            agent.current_node = agent.root
            agent_list.append(agent)

        mcts_iteration = 0
        while True:
            mcts_iteration += 1
            batch_prompts = []
            for agent in agent_list:
                agent_batch_prompts = agent.get_prompts_to_be_expanded(verbose=self.visualize_debug)
                if not agent.is_terminal():
                    assert len(agent_batch_prompts) > 0, len(agent_batch_prompts)
                batch_prompts.extend(agent_batch_prompts)

            # end condition judgement
            if torch.distributed.is_initialized():
                # only account the finish flags for the first rank in each tp group
                if self.tp_rank == 0:
                    local_is_finished = all(agent.is_terminal() for agent in agent_list)
                else:
                    local_is_finished = False

                # Broadcast within TP group first to ensure consistency
                # torch.distributed.barrier()
                [is_finished] = broadcast_pyobj(
                    data=[local_is_finished],
                    rank=self.global_rank,
                    dist_group=self.tp_mesh.get_group(),
                    src=self.tp_mesh.mesh[0].item(),
                    force_cpu_device=False,
                )

                is_finished = torch.tensor([1 if is_finished else 0], dtype=torch.long, device='cuda')
                torch.distributed.all_reduce(is_finished, op=torch.distributed.ReduceOp.SUM)

                if self.visualize_debug and self.tp_rank == 0 and mcts_iteration % 50 == 1:
                    logger.info(
                        f"[RANK {self.global_rank}] is_finished local={local_is_finished}, "
                        f"global sum={is_finished.item()}, threshold={self.global_size}"
                    )

                is_finished = is_finished.item() >= self.global_size
            else:
                is_finished = all(agent.is_terminal() for agent in agent_list)

            # backup & selection may be done inside get_prompts_to_be_expanded, so additional end condition here
            if is_finished:
                if self.visualize_debug and self.tp_rank == 0 and mcts_iteration % 50 == 1:
                    logger.info(
                        f"[RANK {self.global_rank} TP-RANK {self.tp_rank}-{self.tp_size}] "
                        f"MCTS completed after {mcts_iteration} iterations"
                    )
                break

            # Synchronize before generation to ensure all ranks participate
            if torch.distributed.is_initialized():
                # For TP groups, we need to check if tp_rank=0 in each group has prompts                                                                                                                                                                                │ │
                # This ensures all ranks in a TP group participate together
                if self.tp_rank == 0:
                    # Only tp_rank=0 reports whether it has prompts
                    local_has_prompts = len(batch_prompts) > 0
                else:
                    # Other ranks in TP group follow tp_rank=0
                    local_has_prompts = False

                # torch.distributed.barrier()
                [local_has_prompts] = broadcast_pyobj(
                    data=[local_has_prompts],
                    rank=self.global_rank,
                    dist_group=self.tp_mesh.get_group(),
                    src=self.tp_mesh.mesh[0].item(),
                    force_cpu_device=False,
                )

                # Globally check if any rank has prompts to generate
                has_prompts = torch.tensor([1 if local_has_prompts else 0], dtype=torch.long, device='cuda')
                torch.distributed.all_reduce(has_prompts, op=torch.distributed.ReduceOp.SUM)
                
                if self.visualize_debug and self.tp_rank == 0 and mcts_iteration % 50 == 1:
                    logger.info(
                        f"[RANK {self.global_rank}] has_prompts local={local_has_prompts}, "
                        f"global sum={has_prompts.item()}, threshold={self.global_size}"
                    )
                
                # If any rank has no prompts while others do, we need to handle this
                if 0 < has_prompts.item() < self.global_size:
                    # Some ranks have finished but others haven't
                    # All ranks must participate in generate_sequences for the barrier to work
                    if not local_has_prompts:
                        # This rank has finished, but needs to participate in the collective
                        # Create a dummy prompt to maintain synchronization
                        dummy_prompt = agent_list[0].root.ori_data
                        batch_prompts = [dummy_prompt]
                        is_dummy = True
                        if self.visualize_debug and mcts_iteration % 50 == 1:
                            logger.info(
                                f"[RANK {self.global_rank} TP-RANK {self.tp_rank}-{self.tp_size}] "
                                f"MCTS iteration {mcts_iteration}: Activate Dummy Run. Waiting for "
                                f"({has_prompts.item()}/{self.global_size})"
                            )
                    else:
                        is_dummy = False
                # all ranks are not finished yet
                elif has_prompts.item() == self.global_size:
                    is_dummy = False
                # all ranks are finished, should not happen since terminal condition is judged above
                else:
                    logger.info(
                        f"[RANK {self.global_rank} TP-RANK {self.tp_rank}-{self.tp_size}] "
                        f"Unexpected state: has_prompts={has_prompts.item()} (0 means all finished)"
                    )
                    raise RuntimeError(f"[RANK {self.global_rank}] Unexpected state: no ranks have prompts but not finished")
            else:
                is_dummy = False

            # left-padding before stack
            assert len(batch_prompts) > 0, len(batch_prompts)
            # input_lens = [item.batch["input_ids"].shape[1] for item in batch_prompts]
            # if min(input_lens) != max(input_lens):
            #     logger.info(f"Left padding the gen_prompts from {min(input_lens)} to {max(input_lens)}")
            batch_prompts = pad_list_data_protos(batch_prompts, left_pad=True, pad_token_id=self.pad_token_id)

            gen_prompts = stack_data_protos(batch_prompts)
            
            # Log every N iterations to track progress
            if self.visualize_debug and mcts_iteration % 50 == 1 and self.tp_rank == 0 and not is_dummy:
                logger.info(
                    f"[RANK {self.global_rank} TP-RANK {self.tp_rank}-{self.tp_size}] "
                    f"MCTS iteration {mcts_iteration}: generating for {len(gen_prompts.batch)} prompts"
                )
            
            outputs = self.sglang_rollout.generate_sequences(gen_prompts, max_new_tokens=self.step_response_length, **kwargs)
            
            # Skip processing if this was a dummy call
            if is_dummy:
                continue

            # outputs.batch:
            # - prompts: (b x n, 2048), left-padded input prompts for the current rollout step
            # - responses: (b x n, 256), right-padded output responses of the current rollout step
            # - input_ids: (b x n, 2048 + 256), concatenated prompt-response
            # - rollout_log_probs: (b x n, 256), log-probabilities of the current response
            # - attention_mask: (b x n, 2048 + 256), align with input_ids (0 for pad, 1 for valid)
            # - position_ids: (b x n, 2048 + 256), align with input_ids
            # outputs.non_tensor_batch
            # - uid: [b x n,]
            # - reward_model: [b x n,]
            tmp_prompts = outputs.batch.pop("prompts")
            tmp_responses = outputs.batch.pop("responses")
            # truncate the delta parts from the current outputs
            for key in ["input_ids", "attention_mask", "position_ids", "rollout_log_probs"]:
                if outputs.batch[key].shape[1] == tmp_prompts.shape[1] + tmp_responses.shape[1]:
                    # outputs.batch[key] = outputs.batch[key][:, tmp_prompts.shape[1]:]
                    outputs.batch[key] = outputs.batch[key].narrow(1, tmp_prompts.shape[1], tmp_responses.shape[1])
                assert outputs.batch[key].shape[1] == tmp_responses.shape[1]
            del tmp_prompts, tmp_responses

            # expansion
            for agent in agent_list:
                agent_mask = outputs.non_tensor_batch["uid"] == agent.uid
                # record the new children for the current node to be expanded
                if agent_mask.sum() > 0:
                    agent_outputs = mask_select_from_data_proto(outputs, valid_mask=agent_mask)
                    agent.expand_node(agent.current_node, agent_outputs)

                    for child in agent.current_node.children[-self.expansion_width:]:
                        if child.is_terminal:
                            continue
                        has_eos = child.delta_data.batch["attention_mask"][0][-1] == 0
                        if has_eos or agent.current_node.depth == self.max_depth - 1:
                            child.is_terminal = True
                # terminated agent or all children of the current node have already been expanded
                # - for terminated agents, do nothing
                # - for the agent with already-generated children, the expansion has already been done in get_prompts_to_be_expanded()

            # backup
            for agent in agent_list:
                agent.backup_and_selection()

            # fresh memory after each MCTS exploration
            torch.cuda.empty_cache()

        # Record MCTS end time and total iterations
        mcts_end_time = time.time()
        mcts_duration = mcts_end_time - mcts_start_time

        # Store timing info and iterations in each agent for visualization
        for agent in agent_list:
            agent.mcts_iterations = mcts_iteration
            agent.mcts_start_time = mcts_start_time
            agent.mcts_end_time = mcts_end_time
            agent.mcts_duration = mcts_duration

        # broadcasting terminated_prompts after stack by broadcast_pyobj from tp_rank 0 to other tp_rank
        if self.tp_rank == 0:
            terminated_prompts = []
            total_terminated_nodes = 0
            for agent in agent_list:
                agent_terminated_prompts = agent.terminated_prompts
                for term_node, term_prompt in agent_terminated_prompts:
                    if self.use_step_wise_score:
                        # NOTE: extract the FINAL values from the MCTS tree
                        estimated_values = get_values_from_node(term_node)[1:]  # exclude the root
                        estimated_values = list(estimated_values) # make a copy in case
                        term_prompt.batch["rm_scores"] = torch.zeros(
                            term_prompt.batch["responses"].shape,
                            dtype=term_prompt.batch["rollout_log_probs"].dtype,
                            device=term_prompt.batch["rollout_log_probs"].device
                        )
                        assert term_prompt.batch["rm_scores"].shape[1] == len(estimated_values) * self.step_response_length

                        if self.adv_temperature > 0:
                            for i in range(len(estimated_values) - 1):
                                estimated_values[i] = math.tanh(estimated_values[i] / self.adv_temperature) * self.q_value_max

                        overlong_reward = 0.0
                        if self.overlong_buffer_cfg.enable:
                            overlong_buffer_len = self.overlong_buffer_cfg.len
                            expected_len = self.response_length - overlong_buffer_len
                            valid_response_length = term_prompt.batch["attention_mask"][0, self.prompt_length:].sum().item()

                            exceed_len = valid_response_length - expected_len
                            overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)

                        for node_idx, node_value in enumerate(estimated_values):
                            seq_start = node_idx * self.step_response_length
                            seq_end = seq_start + self.step_response_length
                            term_prompt.batch["rm_scores"][:, seq_start:seq_end] = node_value + overlong_reward

                    terminated_prompts.append(term_prompt)
                    total_terminated_nodes += 1
            
            logger.info(
                f"[RANK {self.global_rank} TP-RANK {self.tp_rank}-{self.tp_size}] "
                f"Collected {total_terminated_nodes} terminated nodes from {len(agent_list)} agents"
            )

            terminated_prompts = pad_list_data_protos(
                terminated_prompts, pad_token_id=self.pad_token_id, left_pad=False,
                max_seq_len_libs=dict(
                    responses=self.response_length,
                    rollout_log_probs=self.response_length,
                    rm_scores=self.response_length,
                    input_ids=self.prompt_length + self.response_length,
                    attention_mask=self.prompt_length + self.response_length,
                    position_ids=self.prompt_length + self.response_length,
                    prompts=self.prompt_length
                )
            )

            terminated_prompts = stack_data_protos(terminated_prompts)

        else:
            terminated_prompts = None

        # synchronize between all ranks in a single TP group
        torch.distributed.barrier()
        [terminated_prompts] = broadcast_pyobj(
            data=[terminated_prompts],
            rank=self.global_rank,
            dist_group=self.tp_mesh.get_group(),
            src=self.tp_mesh.mesh[0].item(),
            force_cpu_device=False,
        )

        return terminated_prompts
    
    def __getattr__(self, name):
        return getattr(self.sglang_rollout, name)
