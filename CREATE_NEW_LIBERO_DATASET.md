1) use LIBERO repo git@github.com:OliEfr/LIBERO.git to record new data:
    1.a) run 
            sudo $(which python) scripts/collect_demonstration.py \
                        --controller OSC_POSE \
                        --camera agentview --robots Panda \
                        --num-demonstration 50 \
                        --rot-sensitivity 1.5 \
                        --bddl-file ./libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl
    1.b) run
            python scripts/create_dataset.py --demo-file demonstration_data/robosuite_ln_libero_living_room_tabletop_manipulation_1767090182_7825105_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate/demo.hdf5
        with args.use_camera_obs = False to visually inspect the recorded episodes.
        Note down for which episodes fail to exclude them from the dataset

        Then run, the same script again but with args.use_camera_obs = True and set episodes_to_skip to create the dataset with healty episodes.
    1.c) for postprocessing similar to openvla and pi, run scripts/regenerate_libero_dataset.py
            you can choose to use "is_noop" to remove noop frames.
            you can use "only_render" to only vizualize the episodes
2) Then, you must conver the the libero hdf5 to lerobot format:
    Use the lerobot repo git@github.com:OliEfr/lerobot.git.
    Use libero_h5_minimal.py -> this outputs correct lerobot format, and pushes to hub

    
