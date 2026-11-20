# Offline natural-language critic

The first experiment trains a separate critic on fixed, hindsight-informed
critiques of DeepMath reasoning segments. The actor and target generator stay
frozen. This implements the initial supervised critic experiment (Option 1);
actor updates and bootstrapped critic targets are outside its scope.

The **student** is the trainable critic. The **teacher** is the frozen model
that generates its critique targets. The **actor** is the frozen solver that
produced the cached mathematical reasoning traces.

The "current segment" is a consecutive group of actor reasoning paragraphs;
"preceding reasoning" means all actor reasoning before that group. For
example, when critiquing merged paragraphs 5–7, the student receives the
problem, actor paragraphs 1–4, and paragraphs 5–7. Its task is to critique
the group 5–7.

| Prompt content | Student critic | Teacher / target generator |
| --- | --- | --- |
| Critique instructions and problem statement | Yes | Yes |
| Preceding actor reasoning and current segment | Yes | Yes |
| Remaining actor rollout, including its final answer | No | Yes |
| Verifier score, extracted answer, and reference final answer | No | Yes |

The teacher prompt is built from the student prompt, then extended with the
privileged information and instructions for using it. The teacher is asked to
distinguish correctness from usefulness and express downstream effects as
predictions in its final `<critique>...</critique>`.

During SFT, the student receives its prompt as context and is trained to predict
the teacher's final critique (including its tags and the end-of-sequence token).
Prompt tokens have loss labels of `-100`: they remain visible to the student,
but predicting those tokens is not part of the training objective. Only the
student's parameters are updated; the teacher is used offline to create targets.

Teacher thinking was disabled with `--no_thinking` for the Qwen3-1.7B run and
the MiniCPM thinking-off control. The initial MiniCPM run used thinking, and
the target-generation CLI still enables it by default. When teacher thinking
is enabled, its thinking text is discarded before constructing the student's
training example; only the final critique is used. Student critic prompts use
thinking disabled for both training and prediction.

## Defaults

| Setting | Value |
| --- | --- |
| Actor, target generator, initial critic | `openbmb/MiniCPM5-2B-SFT` |
| Dataset | Shared pinned DeepMath revision and deduplicated holdout |
| Problems | 200 training / 50 validation, disjoint by question |
| Actor rollouts | 2 per question; temperature 1, top-p 0.95, top-k disabled |
| Actor generation limit | 8,192 tokens |
| Segments | Merge paragraphs to at least 50 words, then uniformly sample up to 4 per eligible rollout |
| Target generation | Thinking enabled; 2,048 tokens; temperature 0.6 |
| Critic generation | Thinking disabled; 512 tokens; greedy |
| Generation context limit | 16,384 tokens, including generation budget |
| Training | Full parameters, BF16, completion-only SFT, one epoch |
| Learning rate / schedule | 1e-5 / constant |
| Optimizer / batch / accumulation | 8-bit AdamW / 1 / 16 |
| Training context limit | 12,288 tokens |

Blank lines (including whitespace-only lines) separate nonempty blocks inside
`<think>...</think>`. The opening tag can be supplied by the model's chat
template. Single newlines stay within a segment. Character offsets reference
the original response; no paragraph text is rewritten. The final answer is part
of the teacher-visible continuation, not a reasoning segment. Prefixes are kept
in full.

Consecutive paragraphs are merged greedily until they contain at least
`--min_segment_words 50` whitespace-separated words (`len(text.split())`).
A paragraph already at or above the minimum stays intact. A final short
remainder is appended to the preceding segment. If the entire reasoning is
shorter than the minimum, that rollout is excluded from annotation and counted
as `reasoning_below_min_words`. Merging preserves every internal blank line
and the original character offsets. Sampling happens after merging.

Pass `--min_segment_words 0` for the original one-paragraph segmentation.
New target files use schema version 2 and record inclusive paragraph ranges,
word counts, and segment indices. Existing saved datasets remain readable;
use a new output directory when generating targets with the new schema.

All raw rollouts are retained. Truncated or unfinished rollouts, verifier errors,
and missing/empty reasoning are excluded from annotation and counted. Wrong
answers remain eligible. Truncated or malformed teacher outputs are retained
but excluded from training. Overlength training examples are excluded whole and
reported; generation refuses to silently truncate context. These filters can
bias which traces reach training, so inspect their counts.

## Run

Run from the repository root in the existing environment, with one free GPU.
The native sampler override below is needed for this machine's CUDA toolkit.
Use a distinct `MASTER_PORT` for concurrent experiments.

```sh
export CUDA_VISIBLE_DEVICES=6
export PYTHONHASHSEED=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=29766

.venv/bin/python -m expts.nlac.collect \
    --output_dir data/nlac/minicpm-pilot/corpus

.venv/bin/python -m expts.nlac.targets \
    --corpus_dir data/nlac/minicpm-pilot/corpus \
    --output_dir data/nlac/minicpm-pilot/targets
```

For an initial smoke test, add `--num_train_problems 4 --num_val_problems 2` to
collection and use a separate output directory. `--model_name Qwen/Qwen3-4B`
selects the other model. Model revisions are resolved to commit SHAs and saved;
`--model_revision <sha>` pins one explicitly. The target generator and initial
critic inherit the collected actor's revision unless overridden.

Pass `--no_thinking` to target generation to disable thinking in the model's
chat template. For example, compare Qwen3-1.7B on the same cached MiniCPM
rollouts without recollecting them, using the 50-word minimum:

```sh
.venv/bin/python -m expts.nlac.targets \
    --corpus_dir results/nlac/smoke-20261120/corpus \
    --model_name Qwen/Qwen3-1.7B --no_thinking --min_segment_words 50 \
    --output_dir results/nlac/min50-20261120/qwen3-1.7b-targets
```

The thinking flag is saved with the run settings; changing it requires a new
output directory. The same strict critique parser applies, but a closing
`</think>` tag is only required when the target generator was asked to think.
Critic prediction already disables thinking.

Inspect `targets/preview.txt` and both stages' summary files before scaling up.
The preview contains up to 30 annotations with their visible context. Check a
mix of correct/incorrect rollouts and early/late segments for mathematical
accuracy, usefulness, and references to privileged future information. Parsing
the output format does not establish target quality or prevent semantic leakage.

Save the initial critic's responses on the fixed validation segments, train,
then save the trained responses on those same segments:

```sh
.venv/bin/python -m expts.nlac.predict \
    --targets_dir data/nlac/minicpm-pilot/targets \
    --output_dir results/nlac/minicpm-pilot/baseline

.venv/bin/python -m expts.nlac.train \
    --targets_dir data/nlac/minicpm-pilot/targets \
    --output_dir models/nlac/minicpm-pilot

.venv/bin/python -m expts.nlac.predict \
    --targets_dir data/nlac/minicpm-pilot/targets \
    --model_name models/nlac/minicpm-pilot/final \
    --output_dir results/nlac/minicpm-pilot/trained
```

Prediction includes all selected validation segments, even when their teacher
target failed. This keeps the comparison independent of target acceptance.
Each prediction directory has `responses.txt` and raw JSONL outputs keyed by
segment ID. Validation loss measures agreement with teacher text; it is not a
critic-quality score. A separate response evaluation protocol remains to be
defined.

## Artifacts and restarts

Each generation command writes its configuration before sampling and commits
one JSONL shard per batch. Repeating the same command skips finished shards.
Changing settings, package versions, or source data requires a new output
directory. An interrupted batch may be sampled again. Seeds are fixed per
example, though hardware and library changes can still affect numerical results.

| Artifact | Contents |
| --- | --- |
| `corpus/run_config.json` | Model SHA, sampling settings, dataset/verifier metadata, versions, chat template |
| `corpus/split_manifest.json` | Copy of the shared dataset split |
| `corpus/problems.jsonl` | Question IDs, splits, reference answers, original chat prompts and token IDs |
| `corpus/rollouts/part-*.jsonl` | Sample IDs, text, token IDs, stop reasons, seeds, extracted answers and verifier results |
| `targets/segments.jsonl` | Selected rollout IDs, paragraph ranges, word counts, exact character offsets and selection seeds |
| `targets/targets/part-*.jsonl` | Raw teacher generations, final critiques and rejection reasons |
| `targets/selection_summary.json`, `targets/summary.json` | Selection exclusions and target acceptance counts |
| `training/data_report.json` | Included segment IDs/lengths and exclusions |
| `training/train_summary.json`, `training/log_history.json` | Training metrics and loss history, including validation |
| `training/checkpoint-*`, `training/final` | Resumable trainer checkpoints and final model/tokenizer |

The target config fingerprints the source corpus and selected segments.
Training fingerprints the target files and rejects changes on resume. Source
directories are referenced by absolute path; retain them alongside the run.
Keep local checkpoint directories immutable, since their paths are model
identifiers rather than Hub commit SHAs.

Training supports one process and one visible GPU. Resume explicitly:

```sh
.venv/bin/python -m expts.nlac.train \
    --targets_dir data/nlac/minicpm-pilot/targets \
    --output_dir models/nlac/minicpm-pilot \
    --resume_from_checkpoint models/nlac/minicpm-pilot/checkpoint-100
```

Use the same training settings. With the default constant schedule, `max_steps`
can be extended. For a short integration run, set `--max_steps 1
--gradient_accumulation_steps 1 --save_strategy no`; it still writes `final/`.

## Checks

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    .venv/bin/python -m unittest discover -s tests -p test_nlac.py -v
```

These tests cover exact segment offsets, minimum-size merging and short tails,
exclusions, prompt isolation, final
critique extraction, cached restarts, loss masking, changed-data rejection, and
real training/checkpoint resume with a tiny CPU model. Generation is mocked.

## Initial smoke test

Artifacts are in `results/nlac/smoke-20261120/`; the one-step checkpoint is in
`models/nlac/smoke-20261120/final/`. This run used the defaults except for four
training questions, two validation questions, unmerged paragraphs, and one optimizer step with
gradient accumulation 1. The 57-test repository suite passed, including the
tiny-model training and resume test.

| Stage | Result |
| --- | --- |
| Actor collection | 12 rollouts; 7 completed with correct final answers; 5 truncated |
| Segment selection | 12 training / 15 validation segments |
| Target format acceptance | 12/27 (5 training / 7 validation) |
| Training | One full-parameter step completed and the saved model loaded for generation |
| Baseline / trained response format acceptance | 14/15 / 15/15 on the same validation segments |
| Cached restarts | Targets and baseline responses reused without a visible GPU |

This only verifies the pipeline. The target audit found substantive problems:
many outputs omitted critique tags, and an accepted target (`87591:0:27`)
explicitly cited the verifier and reference answer. Some critiques also assess
more than the current segment. All completed actor rollouts in this tiny sample
were correct, so it does not test judgments of failed solutions. No larger
training run was launched. Review the saved targets before scaling up; format
acceptance and teacher-text loss do not measure critique quality.

## Qwen3-1.7B critique comparison

On the same 27 cached MiniCPM segments, with identical prompts, seeds, sampling
settings, and token limits:

| Target generator | Thinking | Format accepted | Truncated |
| --- | --- | --- | --- |
| MiniCPM5-2B-SFT | On | 12/27 | 0 |
| MiniCPM5-2B-SFT | Off | 17/27 | 2 |
| Qwen3-1.7B | Off | 27/27 | 0 |

Qwen's formatting is reliable in this sample, but content remains problematic.
Five critiques explicitly mention the reference, verifier, or privileged
continuation. Many describe the whole solution instead of the selected
paragraph: for example, `4732:0:2` only says “Now we solve equation.”, but its
critique credits it with a substitution, polynomial derivation, and solution.
Both models show this scope problem. This comparison tests target generation;
the actor rollouts remain MiniCPM's, and no additional training was performed.

The matched outputs and configurations are saved under
`results/nlac/smoke-20261120/`. See `critique-comparison.md` for all three
critiques per segment and `critique-comparison.json` for the counts. The new
runs are `qwen3-1.7b-targets-no-thinking/` and `minicpm-targets-no-thinking/`.

## Minimum 50-word experiment

Greedy merging reduces the same 160 eligible paragraphs to 58 segments. Every
segment meets the minimum: word counts have mean 88.1, median 74, minimum 50,
and maximum 287. Sampling up to four per rollout selects 25 segments, with mean
100.0, median 83, minimum 52, and maximum 287 words.

With thinking off, Qwen3-1.7B produces valid critique formatting on 25/25 and
MiniCPM5-2B-SFT on 18/25. Spot checks still show both models crediting the
current segment with later work and missing local errors. For example,
`4732:0:0` concludes an incorrect domain bound; both teachers approve it and
discuss a substitution and polynomial solution that occur later. The larger
segments remove tiny fragments but do not resolve this observed failure.

The new samples differ from the earlier 27 unmerged paragraphs, so format
counts are descriptive rather than a matched-segment quality comparison.
All 59 repository tests pass. Full outputs, size statistics, and inspection
notes are in `results/nlac/min50-20261120/comparison.md`; raw artifacts and
generation settings are in the adjacent model directories.
