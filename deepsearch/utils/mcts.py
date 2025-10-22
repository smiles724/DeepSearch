import math

from dataclasses import field, dataclass
from typing import Optional, List, Callable, Dict

import torch

from deepsearch.utils.data_proto import slice_from_data_proto, concat_data_protos
from verl import DataProto


@dataclass
class MCTSNode:
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    depth: int = 0
    tag: str = "0"

    ori_data: DataProto = None
    delta_data: DataProto = None
    is_terminal: bool = False
    final_backup: bool = False
    backup_idx: int = -1

    full_prompt: DataProto = None
    token_level_entropy: float = None
    final_score: float = None

    visit_count: int = 0
    value: float = 0.0

    lambda_puct: float = 2.0
    backup_threshold: float = 0.1
    deterministic_sample: bool = False
    q_correct: float = 0.0

    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value / self.visit_count

    def update(self, value: float, traj_len: Optional[int] = None) -> None:
        self.visit_count += 1
        if value != 0:
            assert traj_len is not None
            gemma_weight = max((self.depth + 1) / traj_len, self.backup_threshold) # traj_len has already been added 1
            new_value = self.value + gemma_weight * value

            if self.value * value >= 0:
                self.value = new_value
            # (self.value > 0 & value < 0) | (self.value < 0 & value > 0)
            else:
                if value > 0:
                    self.value = gemma_weight * value
                elif self.value > 0:
                    self.value = self.value
                else:
                    raise RuntimeError((self.value, value))

    def update_recursive(self, value: float, root: 'MCTSNode', traj_len: Optional[int] = None) -> None:
        self.update(value, traj_len=traj_len)
        if self != root and self.parent:
            self.parent.update_recursive(value, root, traj_len=traj_len)

    def puct(self) -> float:
        """Compute PUCT value for selection"""
        if not self.parent:
            return 0
        q = self.q_value() if self.visit_count > 0 else 0
        if self.parent.visit_count == 0 or self.visit_count == 0:
            u = 0
        else:
            u = self.lambda_puct * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        return q + u

    def has_children(self) -> bool:
        return len(self.children) > 0


class MCTSAgent:
    """Manages the MCTS tree for a single prompt"""

    def __init__(
            self,
            data_proto: DataProto,
            pad_token_id: int,
            tokenizer,
            lambda_puct: float,
            expansion_width: int,
            max_depth: int,
            deterministic_sample: bool,
            backtrace_method: str,
            backup_threshold: float,
            q_correct: float,
            step_response_length: int,
            terminate_num: int,
            global_lambda1: float,
            global_lambda2: float,
            global_lambda3: float,
            global_lambda4: float,
            depth_bonus_type: str,
            involve_uncertainty_bonus: bool,
            force_to_frontier: bool,
            reward_fn: Callable,
            incomplete_score: float,
            dynamic_width_config: Optional[Dict] = None,
            ori_raw_prompt_ids: Optional[List[int]] = None,
            logger = None
    ):
        self.root = MCTSNode(
            ori_data=data_proto,
            lambda_puct=lambda_puct,
            deterministic_sample=deterministic_sample,
            backup_threshold=backup_threshold,
            q_correct=q_correct,
        )
        self.data_id = data_proto.non_tensor_batch["index"][0]
        self.uid = data_proto.non_tensor_batch["uid"][0]
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer

        self.current_node = self.root
        self.candidate_nodes = []  # Nodes to be evaluated
        self.max_depth = max_depth
        self.expansion_width = expansion_width
        self.lambda_puct = lambda_puct
        self.deterministic_sample = deterministic_sample
        self.ori_raw_prompt_ids = ori_raw_prompt_ids

        self.backtrace_method = backtrace_method
        self.backup_threshold = backup_threshold
        self.q_correct = q_correct
        self.step_response_length = step_response_length
        self.terminate_num = terminate_num
        self.global_lambda1 = global_lambda1
        self.global_lambda2 = global_lambda2
        self.global_lambda3 = global_lambda3
        self.global_lambda4 = global_lambda4
        self.depth_bonus_type = depth_bonus_type
        self.involve_uncertainty_bonus = involve_uncertainty_bonus
        self.force_to_frontier = force_to_frontier
        self.reward_fn = reward_fn
        self.incomplete_score = incomplete_score
        self.dynamic_width_config = dynamic_width_config

        self.terminated_prompts = []
        self.logger = logger

    def get_prompts_to_be_expanded(self, verbose: bool = False) -> List[DataProto]:
        if self.is_terminal():
            return []

        if self.current_node is None or self.current_node.is_terminal:
            raise RuntimeError(
                f"Node should not be None or terminated at the beginning of each step! "
                f"But got {self.current_node}"
            )

        curr_expansion_width = self.get_expansion_width_at_depth(self.current_node.depth)
        curr_children_num = len(self.current_node.children)
        if curr_children_num < curr_expansion_width:
            # simulation
            full_prompt = get_full_prompt_from_node(self.current_node)
            batch_prompts = [full_prompt] * (curr_expansion_width - curr_children_num)
        else:
            # ONLY the terminated leaf nodes that do not finish score backup should be retained as candidates
            self.candidate_nodes = [child for child in self.current_node.children if not child.final_backup]
            self.backup_and_selection()
            batch_prompts = self.get_prompts_to_be_expanded(verbose=verbose)

        return batch_prompts

    def get_expansion_width_at_depth(self, depth):
        if self.dynamic_width_config:
            min_width = self.dynamic_width_config.get('min_width', 1)
            decay_factor = self.dynamic_width_config.get('decay_factor', 1)

            step_size = self.max_depth // self.expansion_width
            step_number = depth // step_size

            width_reduction = step_number * decay_factor
            width = max(min_width, self.expansion_width - width_reduction)
            return width
        return self.expansion_width

    def is_terminal(self) -> bool:
        return len(self.terminated_prompts) >= self.terminate_num

    def backup_and_selection(self):
        if self.is_terminal():
            return

        reach_terminal = False
        nodes_to_be_backpropagated = []
        for candidate in self.candidate_nodes:
            if candidate.is_terminal:
                cand_traj_len = candidate.depth + 1  # prevent the gemma weight of the leaf node to be 0
                reach_terminal = True

                if candidate.full_prompt is None:
                    candidate_prompt = get_full_prompt_from_node(candidate)
                    mask = (candidate_prompt.batch["input_ids"][0] != self.pad_token_id)
                    left_idx = mask.to(torch.uint8).argmax().item()
                    reversed_mask = mask.flip(dims=[0])
                    right_pad_count = reversed_mask.to(torch.uint8).argmax().item()
                    right_idx = len(mask) - right_pad_count if right_pad_count > 0 or mask[-1] else len(mask)

                    candidate_token_ids = candidate_prompt.batch["input_ids"][0][left_idx:right_idx].tolist()
                    candidate_text = self.tokenizer.decode(candidate_token_ids)
                    ground_truth = candidate_prompt.non_tensor_batch["reward_model"][0]["ground_truth"]
                    reward_dict = self.reward_fn(solution_str=candidate_text, ground_truth=ground_truth)

                    has_eos = candidate_prompt.batch["attention_mask"][0][-1] == 0
                    parsed_ans = reward_dict["pred"]
                    correct = reward_dict["acc"]
                    if self.current_node.depth == self.max_depth - 1 and (not has_eos) and parsed_ans == "[INVALID]":
                        final_value = self.incomplete_score
                    elif correct:
                        final_value = 1.0
                    else:
                        final_value = -1.0

                    candidate_prompt.batch["prompts"] = self.root.ori_data.batch["input_ids"]
                    candidate_prompt.batch["responses"] = (
                        candidate_prompt.batch["input_ids"][:, candidate_prompt.batch["prompts"].shape[1]:]
                    )

                    # calculate token-level entropy for the whole trajectory
                    with torch.no_grad():
                        candidate_log_probs = candidate_prompt.batch["rollout_log_probs"]
                        if right_pad_count > 0:
                            candidate_log_probs = candidate_log_probs[:, :-right_pad_count]
                        if len(candidate_log_probs.shape) == 3:
                            probs = candidate_prompt.batch["rollout_log_probs"].exp()
                            token_level_entropy = torch.special.entr(probs).sum(dim=-1).mean().item()
                        elif len(candidate_log_probs.shape) == 2:
                            # only the probability of tokens selected in the responses are returnd
                            # an approximation to token-level entropy, but OK for heuristic filtering low-confidence
                            token_level_entropy = -1 * candidate_log_probs.mean().item()
                        else:
                            raise RuntimeError(len(candidate_log_probs.shape))

                        # TODO: Entropy may be NaN!!!
                        if math.isnan(token_level_entropy):
                            len_before_trunc = candidate_prompt.batch["rollout_log_probs"].shape[1]
                            log_msg = (
                                f"{self.data_id} meets NaN token_level_entropy from {candidate_log_probs.tolist()} "
                                f"right_pad_count={right_pad_count} len_before_trunc={len_before_trunc}"
                            )
                            if self.logger is not None:
                                self.logger.info(log_msg)
                            else:
                                print(log_msg)

                    candidate.full_prompt = candidate_prompt
                    candidate.token_level_entropy = token_level_entropy
                    candidate.final_score = final_value

                else:
                    candidate_prompt = candidate.full_prompt
                    token_level_entropy = candidate.token_level_entropy
                    final_value = candidate.final_score
                    assert token_level_entropy is not None
                    assert final_value is not None

                nodes_to_be_backpropagated.append(
                    (candidate, final_value, cand_traj_len, candidate_prompt, token_level_entropy)
                )

            else:
                # final_value = 0.0 and traj_len = None for intermediate nodes
                nodes_to_be_backpropagated.append((candidate, 0.0, None))

        intermediate_nodes = [bp_node for bp_node in nodes_to_be_backpropagated if len(bp_node) == 3]
        terminated_nodes = [bp_node for bp_node in nodes_to_be_backpropagated if len(bp_node) == 5]

        # pick up one terminated candidate to bp if there are multiple ones in this step
        if terminated_nodes:
            cand_final_values = [bp_node[1] for bp_node in terminated_nodes]
            # the first priority is to retain the candidates with correct scores 1
            if 1 in cand_final_values:
                cand_retain_indices = [
                    cand_idx for cand_idx, cand_value in enumerate(cand_final_values) if cand_value == 1
                ]
            # if no correct, retain the incorrect complete candidates
            elif -1 in cand_final_values:
                cand_retain_indices = [
                    cand_idx for cand_idx, cand_value in enumerate(cand_final_values) if cand_value == -1
                ]
            # otherwise, the incomplete ones
            else:
                cand_retain_indices = list(range(len(cand_final_values)))

            # ONLY pick up the one with the minimal token-level entropy in the retain candidates
            terminated_nodes = [terminated_nodes[cand_idx] for cand_idx in cand_retain_indices]
            cand_entropies = [bp_node[-1] for bp_node in terminated_nodes]
            # min -> most confident, max -> most informative
            min_ent_cand_idx = cand_entropies.index(min(cand_entropies))
            terminated_nodes = [terminated_nodes[min_ent_cand_idx]]

        # filtering logics ONLY applied to terminated nodes, all intermediate nodes should be retained
        nodes_to_be_backpropagated = intermediate_nodes + terminated_nodes  # intermediate nodes go first

        # 5. Backpropagation
        curr_backup_idx = len(self.terminated_prompts)
        for bp_node in nodes_to_be_backpropagated:
            candidate, final_value, cand_traj_len = bp_node[:3]
            self.eval_and_backpropagate(candidate, final_value, traj_len=cand_traj_len)
            if len(bp_node) == 5:
                candidate.final_backup = True
                candidate.backup_idx = curr_backup_idx
                curr_backup_idx += 1

        # record the terminated trajectory AFTER backpropagation to reflect the real q values
        if terminated_nodes:
            if len(terminated_nodes) > 1:
                raise RuntimeError(len(terminated_nodes))

            candidate_prompt = terminated_nodes[0][-2]
            candidate = terminated_nodes[0][0]
            # # IMPORTANT: values should be extracted after the construction of the entire MCTS tree.
            # # The values extracted here are not the final ones!!!!

            self.terminated_prompts.append((candidate, candidate_prompt))

        if self.is_terminal():
            return

        # 6. Selection for next step
        tmp_candidate_nodes = self.candidate_nodes # for debug
        self.candidate_nodes = []

        if not reach_terminal:
            # unidirectional pick up one from the current children
            # self.current_node = self.selection(from_root=False)
            selected_current_node = self.selection(from_root=False)
            if selected_current_node is not None:
                self.current_node = selected_current_node
            else:
                # selected_current_node=None directly mean all children are is_terminal
                assert all([child.is_terminal for child in self.current_node.children]), \
                    [child.is_terminal for child in self.current_node.children]

                # ONLY the terminated leaf nodes that do not finish score backup should be retained as candidates
                self.current_node = self.global_select(force_to_frontier=True)

        else:
            if self.backtrace_method == "uct":
                # backtrace to the root and use UCT again
                self.current_node = self.selection(from_root=True)
            else:
                self.current_node = self.global_select()

        assert self.current_node is not None

    def global_select(self, force_to_frontier=None):
        if force_to_frontier is None:
            force_to_frontier = self.force_to_frontier
        all_non_terminated_nodes = get_all_non_terminal_nodes(root=self.root)
        if force_to_frontier:
            # use < self.expansion_width instead of == 0, leave space for pre-build
            all_non_terminated_nodes = [node for node in all_non_terminated_nodes if len(node.children) < self.get_expansion_width_at_depth(node.depth)]
        selected_node = select_global_best_node(
            node_list=all_non_terminated_nodes,
            # max_depth=max_depth,  # the current max depth or the configured max depth
            max_depth=self.max_depth,
            lambda1=self.global_lambda1,
            lambda2=self.global_lambda2,
            lambda3=self.global_lambda3,
            lambda4=self.global_lambda4,
            depth_bonus_type=self.depth_bonus_type,
            involve_uncertainty_bonus=self.involve_uncertainty_bonus,
        )
        return selected_node

    def selection(self, from_root=False) -> Optional[MCTSNode]:
        """Select a node to expand"""
        start_node = self.root if from_root else self.current_node

        node = start_node
        if node is None:
            return None

        if node.has_children() or node.is_terminal:
            next_node = self.select_child(node)
            if next_node is None:  # All child nodes are terminal
                node.is_terminal = True
            node = next_node

        # it will become easier for all children to be is_terminal=True when expansion_width becomes smaller
        # return None if (node is None or node.is_terminal) else node
        return None if node.is_terminal else node

    def select_child(self, node: MCTSNode) -> Optional[MCTSNode]:
        children_puct_values = []
        for child in node.children:
            if child.is_terminal:
                children_puct_values.append(-float("inf"))
            else:
                puct_value = child.puct()
                children_puct_values.append(puct_value)

        children_puct_values = torch.tensor(children_puct_values)
        if self.deterministic_sample:
            best_child_idx = children_puct_values.argmax().item()
        else:
            children_puct_probs = children_puct_values.softmax(dim=-1)
            best_child_idx = torch.multinomial(children_puct_probs, num_samples=1).item()

        best_child = node.children[best_child_idx]
        return best_child

    def expand_node(self, node: MCTSNode, outputs: DataProto) -> None:
        self.candidate_nodes = [n for n in self.candidate_nodes if not n.final_backup]

        for idx in range(len(outputs.batch)):
            # Create child node
            new_node = MCTSNode(
                parent=node,
                lambda_puct=self.lambda_puct,
                deterministic_sample=self.deterministic_sample,
                depth=node.depth + 1,
                tag=f"{node.tag}.{len(node.children) + 1}",
                delta_data=slice_from_data_proto(outputs, index=idx),
                backup_threshold=self.backup_threshold,
                q_correct=self.q_correct,
            )

            if new_node.depth >= self.max_depth:
                new_node.is_terminal = True

            node.children.append(new_node)
            self.candidate_nodes.append(new_node)

    def eval_and_backpropagate(self, node: MCTSNode, value: float, traj_len: Optional[int] = None) -> None:
        """Evaluate terminal state and backpropagate values"""
        if node.is_terminal:
            # For terminal nodes, use recursive update
            node.update_recursive(value, self.root, traj_len=traj_len)
        else:
            # For intermediate nodes, only update the current node
            node.update(value, traj_len=traj_len)


def get_full_prompt_from_node(node: MCTSNode) -> DataProto:
    path = []
    current = node
    while current:
        assert (current.ori_data is not None) ^ (current.delta_data is not None), current
        # root node possesses the original data proto
        if current.ori_data is not None:
            path.append(current.ori_data)
        # child node possesses the delta data proto
        elif current.delta_data is not None:
            path.append(current.delta_data)
        current = current.parent

    # reverse the list to make order starting from the root to the latest child
    path = path[::-1]
    full_prompt = concat_data_protos(path)
    return full_prompt


def get_values_from_node(node: MCTSNode) -> List[float]:
    path = []
    current = node
    while current:
        assert (current.ori_data is not None) ^ (current.delta_data is not None), current
        path.append(current.value)
        current = current.parent

    # reverse the list to make order starting from the root to the latest child
    path = path[::-1]
    return path


def get_capital_q_values_from_node(node: MCTSNode) -> List[float]:
    path = []
    current = node
    while current:
        assert (current.ori_data is not None) ^ (current.delta_data is not None), current
        path.append(current.q_value())
        current = current.parent

    # reverse the list to make order starting from the root to the latest child
    path = path[::-1]
    return path


def get_all_non_terminal_nodes(root) -> List[MCTSNode]:
    non_terminal_nodes = []
    visited = set()

    def traverse(node):
        if node is None or id(node) in visited:
            return

        visited.add(id(node))

        if hasattr(node, 'is_terminal') and not node.is_terminal:
            non_terminal_nodes.append(node)

        if hasattr(node, 'children') and node.children:
            for child in node.children:
                traverse(child)

    traverse(root)

    return non_terminal_nodes


def select_global_best_node(
        node_list: List[MCTSNode],
        lambda1: float,
        lambda2: float,
        lambda3: float,
        lambda4: float,
        depth_bonus_type: str,
        involve_uncertainty_bonus: bool = True,
        max_depth: Optional[int] = None
):
    best_score = -float("inf")
    best_node = None

    for node in node_list:
        if node.is_terminal:
            continue
        if node.parent is None:
            continue

        quality_potential = lambda1 * math.tanh(node.parent.q_value())
        if lambda4 > 0:
            quality_potential += lambda4 * math.tanh(node.q_value())

        uncertainty_bonus = 0.
        if involve_uncertainty_bonus:
            node_log_probs = node.delta_data.batch["rollout_log_probs"]
            node_attention_mask = node.delta_data.batch["attention_mask"]
            valid_log_probs = node_log_probs[node_attention_mask == 1]
            token_level_entropy = -1 * valid_log_probs.mean().item()
            uncertainty_bonus = lambda2 * token_level_entropy

            if math.isnan(token_level_entropy):
                print(
                    f"[select_global_best_node WARN] meets NaN token_level_entropy from {valid_log_probs.tolist()} "
                    f"valid_num={node_attention_mask.sum()} len_before_trunc={node_log_probs.shape[1]}"
                )

        if depth_bonus_type == "constant":
            depth_bonus = lambda3 * node.depth
        elif depth_bonus_type == "log":
            depth_bonus = lambda3 * math.log(node.depth + 1)
        elif depth_bonus_type == "sqrt":
            assert max_depth is not None
            depth_bonus = lambda3 * math.sqrt(node.depth / max_depth)
        else:
            raise ValueError

        frontier_priority_score = quality_potential + uncertainty_bonus + depth_bonus
        if frontier_priority_score > best_score:
            best_score = frontier_priority_score
            best_node = node

    return best_node
