Single-step math specialization of [the general pseudocode](pseudo.md).
This describes the paper's full actor/critic loop with shared parameters and
one refinement per candidate solution.

The state `s` is a math problem. The action `a` is the complete attempted
solution, including reasoning and a final `\boxed{...}` answer. Every action
terminates the episode. Reward is 1 for a correct, properly formatted answer
and 0 otherwise. The critic outputs `Correctness: Yes/No`, identifying errors
when it says No. There is no `Future` field or continuation prediction.
See [Section 6.2](https://arxiv.org/html/2512.04601v2#S6.SS2) and
[Appendix B.1](https://arxiv.org/pdf/2512.04601v2#page=17).

Reconstruction detail: v2 prints the math critic prompt, but does not provide
math-specific target-generation and refinement prompts. Below, the terminal
backup uses the observed reward to construct a correctness assessment. The
reference answer is used by the verifier; exposing it to the target generator
would be an additional design choice. Prompt helpers describe their intended
behavior rather than reproduce exact paper prompts.

Functions:

```text
ACT(parameters, s)
    # Generate a complete solution with a boxed final answer.
    return sample_LLM(parameters, actor_prompt(s))


CRITIQUE(parameters, s, a)
    # Assess the attempted solution without the reward or reference answer.
    # Output: Correctness: Yes/No; if No, identify errors, including formatting.
    return sample_LLM(parameters, critic_prompt(s, a))


MAKE_CRITIC_TARGET(parameters, s, a, r)
    # Terminal specialization of the general backup.
    # The target prompt supplies the observed score and asks for a correctness
    # assessment consistent with it, with an error explanation when r == 0.
    # The explanation is model-generated; the score alone does not verify it.
    # Output has the same Correctness format as CRITIQUE.
    return sample_LLM(parameters, target_prompt(s, a, observed_reward=r))


REFINE(parameters, s, a, critique)
    # Produce a replacement COMPLETE solution to the same problem.
    # It may retain a if judged correct. Finish with a boxed answer.
    # The output should stand alone without references to supplied feedback.
    return sample_LLM(parameters, refinement_prompt(s, a, critique))


NLL(parameters, prompt, target_text)
    # Teacher-forced negative log-likelihood on target tokens only.
    prompt_tokens = tokenize(prompt)
    target_tokens = tokenize(target_text)
    return -sum(
        log_probability(
            parameters,
            target_tokens[j],
            context=prompt_tokens + target_tokens[:j]
        )
        for j in range(len(target_tokens))
    )
```

`rewrite_target_reasoning` below rewrites only the optional reasoning block,
preserving the subsequent correctness assessment or final answer. Its purpose
is to remove references to information absent from the student's input, as in
[Appendices B.2–B.3](https://arxiv.org/pdf/2512.04601v2#page=18).

Algorithm:

```text
INPUTS:
    pretrained_parameters
    math_dataset                   # problem/reference-answer pairs
    verifier                       # binary answer correctness + format check
    number_of_iterations
    solutions_per_iteration
    updates_per_iteration
    batch_size
    critic_optimizer
    actor_optimizer
    model_emits_reasoning_traces
    tau = 0.005


# STEP 1 — Initialize the shared model and replay buffer.
θ = copy(pretrained_parameters)
θ_target = copy(θ)
D = ReplayBuffer()


for iteration in range(number_of_iterations):

    # STEP 2 — Collect complete solutions and terminal rewards.
    for sample in range(solutions_per_iteration):
        s, reference_answer = math_dataset.sample()

        with no_grad():
            a_executed = ACT(θ, s)

        r = verifier.score(a_executed, reference_answer)
        D.add(s=s, a=a_executed, r=r)
        # All entries are terminal: no s_next or continuation is needed.


    for update in range(updates_per_iteration):

        # STEP 3 — Sample previously scored solutions.
        batch = D.sample(batch_size)
        critic_examples = []


        # STEP 4 — Build correctness targets using the observed rewards.
        with no_grad():
            for s, a_old, r in batch:
                y_target = MAKE_CRITIC_TARGET(θ_target, s, a_old, r)

                if model_emits_reasoning_traces:
                    y_target = rewrite_target_reasoning(
                        parameters=θ_target,
                        target=y_target,
                        student_context=(s, a_old),
                        instruction=(
                            "Assess correctness from the problem and solution; "
                            "remove references to the reward as supplied information. "
                            "Preserve the final correctness assessment."
                        )
                    )

                # The student sees the attempted solution, but not its score.
                critic_examples.append((critic_prompt(s, a_old), y_target))


        # STEP 5 — Train the critic behavior of the shared model.
        L_critic = mean(
            NLL(θ, prompt, y_target)
            for prompt, y_target in critic_examples
        )
        critic_optimizer.step(
            parameters=θ,
            gradient=grad(L_critic, θ)
        )
        # This also changes the parameters used by ACT and REFINE.


        # STEP 6 — Generate revised-solution training targets.
        actor_examples = []

        with no_grad():
            for s, a_old, r in batch:
                a_candidate = ACT(θ, s)

                # r belongs to a_old, not to this newly sampled candidate.
                # The critique uses neither r nor the reference answer.
                critique = CRITIQUE(θ, s, a_candidate)
                a_refined = REFINE(θ, s, a_candidate, critique)

                if model_emits_reasoning_traces:
                    a_refined = rewrite_target_reasoning(
                        parameters=θ,
                        target=a_refined,
                        student_context=s,
                        instruction=(
                            "Present a self-contained solution from the problem alone; "
                            "remove references to the earlier attempt and critique. "
                            "Preserve the final answer."
                        )
                    )

                actor_examples.append((actor_prompt(s), a_refined))


        # STEP 7 — Distill revised solutions into the actor.
        # The student receives only the original problem.
        # No verifier-based filtering of revised solutions is added here.
        L_actor = mean(
            NLL(θ, prompt, a_refined)
            for prompt, a_refined in actor_examples
        )
        actor_optimizer.step(
            parameters=θ,
            gradient=grad(L_actor, θ)
        )


        # STEP 8 — Slowly update the target model.
        θ_target = tau * θ + (1 - tau) * θ_target


RETURN:
    actor using ACT(θ, s)
    critic using CRITIQUE(θ, s, a)
```

The update order and EMA follow [Section 5.2](https://arxiv.org/html/2512.04601v2#S5.SS2).
As in `pseudo.md`, replay prioritization and optimizer-state details are left
abstract. Appendix B.4 describes prioritizing replay by critic loss.
