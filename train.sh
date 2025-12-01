# lerobot-train --config_path=config/train_diffusion_pusht_partial.yaml
# lerobot-train --config_path=config/train_diffusion_pusht_visual_coverTee.yaml
# lerobot-train --config_path=config/train_diffusion_pusht_visual.yaml
# lerobot-train --config_path=config/train_diffusion_pusht_visual.yaml



lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.scheduler_decay_steps=200000 \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_10 \
  --steps=200000 \
  --batch_size=32 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval_freq=10000 \
  --wandb.enable=true \
  --wandb.project=policies_lerobot_smolvla \
  --policy.input_features='{"observation.images.image": {"type": "VISUAL", "shape": [3,256,256]}}'

# python src/lerobot/scripts/lerobot_train.py \
#     --dataset.repo_id=HuggingFaceVLA/libero \
#     --policy.type=pi05 \
#     --job_name=pi05_training \
#     --policy.push_to_hub=false \
#     --policy.pretrained_path=lerobot/pi05_libero_base \
#     --policy.compile_model=true \
#     --policy.gradient_checkpointing=true \
#     --policy.dtype=bfloat16 \
#     --wandb.enable=true \
#     --wandb.project=policies_lerobot_pi05 \
#     --policy.dtype=bfloat16 \
#     --steps=6000 \
#     --policy.scheduler_decay_steps=3000 \
#     --policy.device=cuda \
#     --batch_size=16 \
#     --env.type=libero \
#     --env.task=libero_10 \
#     --eval.batch_size=1 \
#     --eval.n_episodes=10 \
#     --eval_freq=1000