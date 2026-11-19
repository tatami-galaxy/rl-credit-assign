"""Explicit visible and privileged inputs for Option 1."""

CRITIC_INSTRUCTION = """Evaluate the CURRENT SEGMENT of a mathematical reasoning trace in the context of the problem and preceding reasoning.
Explain briefly what the segment does, whether its reasoning is justified, and how it may help or hinder further progress. Distinguish mathematical correctness from usefulness: a correct step can be irrelevant, and an unsuccessful attempt can expose a useful constraint. Say when the evidence is insufficient. Do not infer that every segment is bad from an unsuccessful outcome.
Return a concise, self-contained critique (roughly 80-160 words) inside <critique>...</critique>. Discuss possible downstream effects as predictions. Do not rewrite or finish the solution."""

TARGET_INSTRUCTION = """You also receive privileged hindsight: the remainder of this sampled rollout, its verifier result, and the reference final answer. Use these to check your assessment of the CURRENT SEGMENT. The verifier checks final-answer correctness and formatting, not every reasoning step. One observed continuation does not establish expected value or causal responsibility.
The final critique will train a critic that sees ONLY the problem, preceding reasoning, and current segment. Do not refer to supplied hindsight, the reference answer, or the verifier as evidence in that critique. Do not report specific later actions as already observed; express possible consequences as predictions. Keep any private analysis outside the final <critique>...</critique> block."""


def critic_prompt(question, prefix, segment):
    return [
        {"role": "system", "content": CRITIC_INSTRUCTION},
        {"role": "user", "content": (
            f"PROBLEM\n{question}\n\nPRECEDING REASONING\n{prefix}\n\nCURRENT SEGMENT\n{segment}"
        )},
    ]


def target_prompt(question, prefix, segment, continuation, score, model_answer, reference_answer):
    messages = critic_prompt(question, prefix, segment)
    messages[0]["content"] += "\n\n" + TARGET_INSTRUCTION
    messages[1]["content"] += (
        f"\n\nPRIVILEGED REMAINDER OF ROLLOUT\n{continuation}"
        f"\n\nVERIFIER SCORE\n{score}"
        f"\n\nEXTRACTED FINAL ANSWER\n{model_answer}"
        f"\n\nREFERENCE FINAL ANSWER\n{reference_answer}"
    )
    return messages
