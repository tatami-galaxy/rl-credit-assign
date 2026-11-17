Functions : 

```
ACT(parameters, s)
    # Generate an action from the interaction history.
    return sample_LLM(parameters, actor_prompt(s))


CRITIQUE(parameters, s, a)
    # Predict a possible continuation after taking a in s.
    # Then assess a using that prediction.
    return sample_LLM(
        parameters,
        critic_prompt(s, a)
    )
    # Output contains:
    #   future: compact prediction through task success/failure
    #   evaluation: assessment of a and suggested improvement


PREDICT_CONTINUATION(parameters, s_next)
    # Predict what the agent and environment will do from s_next.
    # The practical implementation does not separately sample
    # the next actor action before generating this continuation.
    return sample_LLM(
        parameters,
        continuation_prompt(s_next)
    )


MAKE_CRITIC_TARGET(parameters, s, a, r, s_next, continuation)
    # Construct an evaluation of a using:
    #   - the response actually observed after a,
    #   - the observed reward,
    #   - a predicted continuation from that new state.
    #
    # Combine the observed transition and predicted continuation.
    # Give the immediate observation greater emphasis.
    return sample_LLM(
        parameters,
        backup_prompt(s, a, r, s_next, continuation)
    )
    # Output has the same format as CRITIQUE:
    #   future + evaluation


REFINE(parameters, s, a, critique)
    # Generate a replacement for a at the SAME state s.
    # It may retain a if the evaluation judges it satisfactory.
    return sample_LLM(
        parameters,
        refinement_prompt(s, a, critique)
    )


NLL(parameters, prompt, target_text)
    # Teacher-forced negative log-likelihood.
    # Compute loss on target tokens, excluding prompt tokens.
    return -sum(
        log_probability(
            parameters,
            target_text[j],
            context=prompt + target_text[:j]
        )
        for j in range(len(target_text))
    )
```

Algo : 

```
INPUTS:
    pretrained_parameters
    environment
    number_of_iterations
    collection_steps_per_iteration
    updates_per_iteration
    batch_size
    critic_optimizer
    actor_optimizer
    tau = 0.005


# ============================================================
# STEP 1 — Initialize the shared model and replay buffer
# ============================================================

θ = copy(pretrained_parameters)
θ_target = copy(θ)

D = ReplayBuffer()
s_live = environment.reset()


for iteration in range(number_of_iterations):

    # ========================================================
    # STEP 2 — Collect real environment transitions
    # ========================================================

    for step in range(collection_steps_per_iteration):

        with no_grad():
            a_executed = ACT(θ, s_live)

        observation, r, terminal = environment.step(
            environment_action_part(a_executed)
        )

        s_next = append_interaction(
            s_live,
            environment_action_part(a_executed),
            observation
        )

        D.add(
            s=s_live,
            a=a_executed,
            r=r,
            s_next=s_next,
            terminal=terminal
        )

        if terminal:
            s_live = environment.reset()
        else:
            s_live = s_next


    for update in range(updates_per_iteration):

        # ====================================================
        # STEP 3 — Sample previously collected transitions
        # ====================================================

        batch = D.sample(batch_size)

        # These transitions can come from older actor versions.
        critic_examples = []


        # ====================================================
        # STEP 4 — Build language Bellman targets
        # ====================================================

        with no_grad():

            for s, a_old, r, s_next, terminal in batch:

                if terminal:
                    # Added terminal bookkeeping:
                    # do not invent a continuation after termination.
                    continuation = END_OF_EPISODE
                else:
                    continuation = PREDICT_CONTINUATION(
                        θ_target,
                        s_next
                    )

                y_target = MAKE_CRITIC_TARGET(
                    θ_target,
                    s,
                    a_old,
                    r,
                    s_next,
                    continuation
                )

                if model_emits_reasoning_traces:
                    y_target = rewrite_target_reasoning(
                        parameters=θ_target,
                        target=y_target,
                        student_context=(s, a_old),
                        instruction=(
                            "Express future outcomes as predictions; "
                            "remove references to privileged observations "
                            "or rewards as information already supplied."
                        )
                    )

                critic_examples.append(
                    (critic_prompt(s, a_old), y_target)
                )


        # ====================================================
        # STEP 5 — Train the critic behavior
        # ====================================================

        L_critic = mean(
            NLL(θ, prompt, y_target)
            for prompt, y_target in critic_examples
        )

        critic_optimizer.step(
            parameters=θ,
            gradient=grad(L_critic, θ)
        )

        # This changes the shared LLM parameters.
        # It can therefore also affect the actor and other roles.


        # ====================================================
        # STEP 6 — Generate improved-action training targets
        # ====================================================

        actor_examples = []

        with no_grad():

            for s, a_old, r, s_next, terminal in batch:

                # 6a. Draw a fresh action from the current actor.
                a_candidate = ACT(θ, s)

                # 6b. Predict its future and critique it.
                # Do not use s_next here: s_next resulted from a_old,
                # which may differ from a_candidate.
                critique = CRITIQUE(
                    θ,
                    s,
                    a_candidate
                )

                # 6c. Generate a replacement action at s.
                a_refined = REFINE(
                    θ,
                    s,
                    a_candidate,
                    critique
                )

                if model_emits_reasoning_traces:
                    a_refined = rewrite_target_reasoning(
                        parameters=θ,
                        target=a_refined,
                        student_context=s,
                        instruction=(
                            "Motivate the action from s alone; "
                            "remove references to the candidate action "
                            "and its critique as supplied feedback."
                        )
                    )

                # The student's input contains only the actor prompt.
                actor_examples.append(
                    (actor_prompt(s), a_refined)
                )


        # ====================================================
        # STEP 7 — Distill refined actions into the actor
        # ====================================================

        L_actor = mean(
            NLL(θ, prompt, a_refined)
            for prompt, a_refined in actor_examples
        )

        actor_optimizer.step(
            parameters=θ,
            gradient=grad(L_actor, θ)
        )


        # ====================================================
        # STEP 8 — Slowly update the target model
        # ====================================================

        θ_target = tau * θ + (1 - tau) * θ_target


RETURN:
    actor using ACT(θ, s)
```