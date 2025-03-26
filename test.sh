#!/bin/bash
source /home/ukenryu/map_test/lanelet2_python_api_for_autoware/setup.bash
python3 test_and_visualize.py \
    --cache_file /home/ukenryu/map_test/lanelet2_python_api_for_autoware/extract_bags/caches/cache.json \
    --map_file /home/ukenryu/autoware_map/odaiba_stable/lanelet2_map.osm \
    --model_path /home/ukenryu/map_test/lanelet2_python_api_for_autoware/naiveTF/outputs/best_model.pth \
    --num_samples 5 \
    --output_dir visualization_outputs \
    --mode interactive