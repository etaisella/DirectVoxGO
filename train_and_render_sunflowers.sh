#!/bin/bash
echo "Starting Edit!"

python run.py --config configs/nerf_360/flowers.py --run_voxe --voxe_prompt "a photo of sunflowers in a vase on an outdoor deck" --render_test

echo "Starting Video render!"

python run.py --config configs/nerf_360/flowers.py  --voxe_prompt "a photo of sunflowers in a vase on an outdoor deck" --render_only --render_video_voxe