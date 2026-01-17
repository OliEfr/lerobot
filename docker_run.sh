export LIBERO_PATH=~/project_repos/LIBERO
docker run -it --gpus all --env-file ./docker/.env \
  -v $(pwd):/lerobot \
  -v ${LIBERO_PATH}/libero/libero/init_files:/libero_data/init_files:ro \
  -v ${LIBERO_PATH}/libero/libero/bddl_files:/libero_data/bddl_files:ro \
  -v ${LIBERO_PATH}/libero/libero/assets:/opt/lerobot-venv/lib/python3.10/site-packages/libero/libero/assets:ro \
  -v $(pwd)/libero_cluster_config.yaml:/home/user_lerobot/.libero/config.yaml:ro \
  --rm lerobot-user