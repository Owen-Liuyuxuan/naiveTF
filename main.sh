#!/bin/bash
source /home/ukenryu/map_test/lanelet2_python_api_for_autoware/setup.bash
python3 main.py --train_cache /home/ukenryu/map_test/lanelet2_python_api_for_autoware/extract_bags/new_map_cache \
    --val_cache /home/ukenryu/map_test/lanelet2_python_api_for_autoware/extract_bags/new_map_cache/cache_532d0885-359b-4274-85bb-39bdb4373457_2025-03-25-15-10-39.json\
    --map_file /home/ukenryu/autoware_map/shinagawa_odaiba_beta/lanelet2_map.osm \
    --epochs 90000  --lr 1e-4