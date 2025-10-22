import argparse
import glob
import json
import os
from os.path import basename, join

from datasets import load_dataset
from tqdm import tqdm

from verl.utils.reward_score.math_dapo import compute_score_boxed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_dir", type=str, required=True)
    parser.add_argument("--buffer_register_threshold", type=float, default=0.25)
    args = parser.parse_args()

    step_jsonl_list = glob.glob(os.path.join(args.rollout_dir, "*.jsonl"))
    step_jsonl_list.sort(key=lambda x: int(basename(x).replace(".jsonl", "")))

    deepmath_103k_ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    question2id = {}
    for idx, doc in enumerate(tqdm(deepmath_103k_ds, desc="registering questions")):
        question = doc["question"]
        if question in question2id:
            continue
        question2id[question] = f"zwhe99--DeepMath-103K-{idx}"

    id2responses = {}
    for step_jsonl_path in tqdm(step_jsonl_list, desc="gathering step jsonl data"):
        with open(step_jsonl_path, "r") as f:
            for line in f.readlines():
                doc = json.loads(line)

                doc_question = doc["input"]
                doc_question = doc_question.removeprefix("<｜User｜>")
                doc_question = doc_question.removesuffix(
                    " Let's think step by step and output the final answer within \\boxed{}.<｜Assistant｜><think>\n")
                doc_id = question2id[doc_question]
                if doc_id not in id2responses:
                    id2responses[doc_id] = []

                doc_response = doc["output"]
                id2responses[doc_id].append(doc_response)

    replay_buffer = {}
    total_questions = len(id2responses)
    total_responses = 0
    total_correct_responses = 0
    questions_with_any_correct = 0

    for doc_id in tqdm(id2responses, desc="gathering replay buffer data"):
        ds_idx = int(doc_id.split("-")[-1])

        doc_responses = id2responses[doc_id]
        total_responses += len(doc_responses)

        doc_correct_responses = []
        for resp in doc_responses:
            reward_dict = compute_score_boxed(
                solution_str=resp,
                ground_truth=deepmath_103k_ds[ds_idx]["final_answer"]
            )
            correct = reward_dict["acc"]
            if correct:
                doc_correct_responses.append(resp)

        if len(doc_correct_responses) > 0:
            questions_with_any_correct += 1
            total_correct_responses += len(doc_correct_responses)

        if 0 < len(doc_correct_responses) / len(doc_responses) <= args.buffer_register_threshold:
            replay_buffer[doc_id] = doc_correct_responses

    print("\n" + "=" * 60)
    print("Statistics:")
    print(f"  Total questions processed: {total_questions}")
    print(
        f"  Questions with any correct response: {questions_with_any_correct} ({questions_with_any_correct / total_questions * 100:.2f}%)")
    print(f"  Total responses: {total_responses}")
    print(
        f"  Total correct responses: {total_correct_responses} ({total_correct_responses / total_responses * 100:.2f}%)")
    print(f"  Questions added to replay buffer: {len(replay_buffer)}")
    print(f"  Correct responses in replay buffer: {sum(len(v) for v in replay_buffer.values())}")
    print(f"  Buffer selection criteria: 0% < accuracy <= {args.buffer_register_threshold * 100:.1f}%")
    print("=" * 60 + "\n")

    with open(join(args.rollout_dir, f"replay_buffer_{args.buffer_register_threshold}.json"), "w") as f:
        json.dump(replay_buffer, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
