lerobot-eval \
    --policy.path=outputs/train/2025-11-27/10-59-52_train_diffusion_pusht_partial/checkpoints/last/pretrained_model \
    --config=config/envs/pusht_coverTee.yaml \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --policy.use_amp=false \
    --policy.device=cuda


# lerobot-eval \
#   --env.type=libero \
#   --env.task=libero_10 \
#   --eval.batch_size=1 \
#   --eval.n_episodes=10 \
#   --policy.path=lerobot/pi05_libero_finetuned \
#   --policy.n_action_steps=10 \
#   --policy.compile_model=true \
#   --policy.dtype=bfloat16 \
#   --env.max_parallel_tasks=1

# lerobot-eval \
#   --env.type=libero \
#   --env.task=libero_object \
#   --eval.batch_size=1 \
#   --eval.n_episodes=10 \
#   --policy.path=lerobot/pi05_libero_finetuned \
#   --policy.n_action_steps=10 \
#   --policy.compile_model=true \
#   --policy.dtype=bfloat16 \
#   --env.max_parallel_tasks=1

# lerobot-eval \
#   --env.type=libero \
#   --env.task=libero_spatial \
#   --eval.batch_size=1 \
#   --eval.n_episodes=10 \
#   --policy.path=lerobot/pi05_libero_finetuned \
#   --policy.n_action_steps=10 \
#   --policy.compile_model=true \
#   --policy.dtype=bfloat16 \
#   --env.max_parallel_tasks=1