#!/bin/bash
source /home/ukenryu/map_test/lanelet2_python_api_for_autoware/setup.bash
python3 test_and_visualize.py \
    --cache_file /home/ukenryu/map_test/lanelet2_python_api_for_autoware/extract_bags/new_map_cache/cache_532d0885-359b-4274-85bb-39bdb4373457_2025-03-25-15-10-39.json \
    --map_file /home/ukenryu/autoware_map/shinagawa_odaiba_beta/lanelet2_map.osm \
    --model_path /home/ukenryu/map_test/lanelet2_python_api_for_autoware/naiveTF/outputs/best_model.pth \
    --num_samples 5 \
    --output_dir visualization_outputs \
    --mode interactive