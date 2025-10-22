from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Dict

import numpy as np
import torch

from verl import DataProto
from verl.utils.torch_functional import pad_sequence_to_length


def slice_from_data_proto(data_proto: DataProto, index: int, stride: int = 1) -> DataProto:
    slice_data = defaultdict()
    for k, v in data_proto.batch.items():
        assert v.dim() >= 2, (k, v.dim())
        slice_data[k] = v[index: index + stride]
    for k, v in data_proto.non_tensor_batch.items():
        slice_data[k] = np.fromiter(v[index: index + stride], dtype=object, count=stride)

    slice_data = DataProto.from_single_dict(slice_data, meta_info=data_proto.meta_info)
    return slice_data


def mask_select_from_data_proto(data_proto: DataProto, valid_mask) -> DataProto:
    slice_data = defaultdict()
    for k, v in data_proto.batch.items():
        assert v.dim() >= 2, (k, v.dim())
        slice_data[k] = v[valid_mask]
    for k, v in data_proto.non_tensor_batch.items():
        slice_data[k] = np.fromiter(v[valid_mask], dtype=object, count=valid_mask.sum())

    slice_data = DataProto.from_single_dict(slice_data, meta_info=data_proto.meta_info)
    return slice_data


def stack_data_protos(data_protos: List[DataProto]) -> DataProto:
    if not data_protos:
        raise ValueError("data_protos list is empty")

    # Collect tensor and non-tensor data separately
    tensor_data = defaultdict(list)
    non_tensor_data = defaultdict(list)

    for dp in data_protos:
        for k, v in dp.batch.items():
            # Ensure 2D tensor (batch_size=1, seq_len)
            if v.dim() == 1:
                v = v.unsqueeze(0)
            tensor_data[k].append(v)
        for k, v in dp.non_tensor_batch.items():
            non_tensor_data[k].extend(v)

    final_batch = {}
    for k, v_list in tensor_data.items():
        # Ensure all tensors are on the same device
        if len(v_list) > 0:
            # Check if any tensor is on cuda
            devices = [v.device for v in v_list]
            cuda_devices = [d for d in devices if d.type == 'cuda']

            if cuda_devices:
                # If there's any cuda device, use the first cuda device found
                target_device = cuda_devices[0]
                # Only move CPU tensors to CUDA, keep CUDA tensors as-is
                v_list = [v.to(target_device) if v.device.type == 'cpu' else v for v in v_list]
            # If no cuda devices, keep all as-is (all should be CPU)

        final_batch[k] = torch.cat(v_list, dim=0)

    final_non_tensor = {}
    for k, v_list in non_tensor_data.items():
        # assert len(v_list) == 1, len(v)
        final_non_tensor[k] = np.fromiter(v_list, dtype=object, count=len(v_list))

    meta_info = None
    for dp in data_protos:
        if dp.meta_info is not None and len(dp.meta_info) > 0:
            meta_info = dp.meta_info
            break
    final_data = DataProto.from_single_dict({**final_batch, **final_non_tensor}, meta_info=meta_info)
    return final_data


def concat_data_protos(data_protos: List[DataProto]) -> DataProto:
    if not data_protos:
        raise ValueError("data_protos list is empty")

    tensor_data = defaultdict(list)
    non_tensor_data = defaultdict(list)

    for dp in data_protos:
        for k, v in dp.batch.items():
            # Ensure 2D tensor (batch_size=1, seq_len)
            if v.dim() == 1:
                v = v.unsqueeze(0)
            tensor_data[k].append(v)
        for k, v in dp.non_tensor_batch.items():
            assert len(v) == 1, len(v)
            if len(non_tensor_data[k]) == 0:
                non_tensor_data[k].extend(v)
            else:
                assert non_tensor_data[k][0] == v[0]

    final_batch = {}
    for k, v_list in tensor_data.items():
        # Ensure all tensors are on the same device
        if len(v_list) > 0:
            # Check if any tensor is on cuda
            devices = [v.device for v in v_list]
            cuda_devices = [d for d in devices if d.type == 'cuda']
            
            if cuda_devices:
                # If there's any cuda device, use the first cuda device found
                target_device = cuda_devices[0]
                # Only move CPU tensors to CUDA, keep CUDA tensors as-is
                v_list = [v.to(target_device) if v.device.type == 'cpu' else v for v in v_list]
            # If no cuda devices, keep all as-is (all should be CPU)
            
        final_batch[k] = torch.cat(v_list, dim=-1)

    final_non_tensor = {}
    for k, v_list in non_tensor_data.items():
        final_non_tensor[k] = np.array([v_list[0]], dtype=object)

    meta_info = None
    for dp in data_protos:
        if dp.meta_info is not None:
            meta_info = dp.meta_info
            break
    final_data = DataProto.from_single_dict({**final_batch, **final_non_tensor}, meta_info=meta_info)
    return final_data


def deduplicate_prompts(prompts: DataProto, repeat_num: Optional[int] = None) -> DataProto:
    if "uid" in prompts.non_tensor_batch.keys():
        batch_uids = prompts.non_tensor_batch["uid"]
    elif "index" in prompts.non_tensor_batch.keys():
        batch_uids = prompts.non_tensor_batch["index"]
    else:
        raise RuntimeError(
            f"Neither 'uid' or 'index' keys are not in prompts! "
            f"available non_tensor_batch keys: {prompts.non_tensor_batch.keys()}"
        )

    uid2batch_idx = {}
    for idx, uid in enumerate(batch_uids):
        if uid not in uid2batch_idx:
            uid2batch_idx[uid] = []
        uid2batch_idx[uid].append(idx)

    for uid, batch_idx_list in uid2batch_idx.items():
        uid2batch_idx[uid].sort()
        if repeat_num is not None:
            assert len(batch_idx_list) == repeat_num, (
                f"The number of the prompt with uid {uid} ({len(batch_idx_list)}) mismatches with the given n "
                f"{repeat_num}"
            )

    unique_batch_idx_list = [batch_idx_list[0] for batch_idx_list in uid2batch_idx.values()]

    unique_prompts_tensors = defaultdict()
    unique_prompts_non_tensors = defaultdict()

    for k, v in prompts.batch.items():
        unique_prompts_tensors[k] = v[unique_batch_idx_list]
    for k, v in prompts.non_tensor_batch.items():
        unique_prompts_non_tensors[k] = v[unique_batch_idx_list]

    unique_prompts = DataProto.from_single_dict(
        {**unique_prompts_tensors, **unique_prompts_non_tensors}, meta_info=prompts.meta_info
    )
    return unique_prompts


def pad_list_data_protos(
        data_protos: List[DataProto],
        pad_token_id: Optional[int] = None,
        left_pad: bool = False,
        max_seq_len_libs: Optional[Dict] = None,
) -> List[DataProto]:
    data_protos = deepcopy(data_protos) # prevent in-place modification
    if pad_token_id is None:
        pad_token_id = data_protos[0].meta_info["pad_token_id"]

    for key in data_protos[0].batch.keys():
        max_seq_len = max(dp.batch[key].shape[1] for dp in data_protos) if max_seq_len_libs is None else max_seq_len_libs[key]

        if key in ["responses", "input_ids", "rollout_log_probs", "attention_mask", "rm_scores", "prompts"]:
            if key in ["rollout_log_probs", "attention_mask", "rm_scores"]:
                current_pad_id = 0
            else:
                current_pad_id = pad_token_id

            for prmt_idx in range(len(data_protos)):
                data_protos[prmt_idx].batch[key] = pad_sequence_to_length(
                    tensors=data_protos[prmt_idx].batch[key],
                    max_seq_len=max_seq_len,
                    pad_token_id=current_pad_id,
                    left_pad=left_pad,
                )

        elif key == "position_ids":
            for prmt_idx in range(len(data_protos)):
                prmt_position_ids = data_protos[prmt_idx].batch["position_ids"]
                if prmt_position_ids.shape[1] < max_seq_len:
                    if left_pad:
                        data_protos[prmt_idx].batch[key] = pad_sequence_to_length(
                            tensors=data_protos[prmt_idx].batch[key],
                            max_seq_len=max_seq_len,
                            pad_token_id=0,
                            left_pad=left_pad,
                        )
                    else:
                        delta_len = max_seq_len - prmt_position_ids.shape[1]
                        base_pos = prmt_position_ids[0, -1].item()
                        delta_position_id = torch.arange(
                            base_pos + 1, base_pos + delta_len + 1,
                            device=prmt_position_ids.device
                        ).expand(prmt_position_ids.shape[0], -1)

                        data_protos[prmt_idx].batch["position_ids"] = (
                            torch.cat([prmt_position_ids, delta_position_id], dim=-1)
                        )

        else:
            raise KeyError(key)

    return data_protos
