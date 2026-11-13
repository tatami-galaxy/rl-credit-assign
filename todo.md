- Change max ent objective?
- NLAC, segment policy opt, multi turn dense feedback
- Iterative response generation

SFT Models : 

- [MiniCPM](https://huggingface.co/openbmb/MiniCPM5-2B-SFT)
- [DS Qwen](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
- [Olmo](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT)

NLAC : 

θ = pretrained_parameters()
θ_target = copy(θ)
replay = []

while training:

    # Collect actual environment transitions.
    replay.extend(rollout(policy=actor(θ)))

    for s, a, reward, s_next, terminal in sample(replay):

        # 1. Policy evaluation: construct a language Bellman target.
        with no_grad():
            future = (
                "" if terminal
                else predict_future(θ_target, s_next)
            )
            target_description = language_backup(
                θ_target, reward, s_next, future
            )

        critic_loss = NLL(
            model=θ,
            prompt=prediction_prompt(s, a),
            target=target_description,
        )
        θ = gradient_step(θ, critic_loss, critic_learning_rate)

        # 2. Policy improvement: critique, refine, then distill.
        with no_grad():
            proposal = actor(θ).sample(s)
            futures = [
                predict_future(θ, s, proposal)
                for _ in range(k)
            ]
            critique = evaluate(θ, s, proposal, futures)
            refined_action = refine(θ, s, proposal, critique)

        actor_loss = NLL(
            model=θ,
            prompt=actor_prompt(s),
            target=refined_action,
        )
        θ = gradient_step(θ, actor_loss, actor_learning_rate)

        # 3. Slowly update the target model.
        θ_target = τ * θ + (1 - τ) * θ_target