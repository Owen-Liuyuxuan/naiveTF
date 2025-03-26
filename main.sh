#!/bin/bash
source /home/ukenryu/map_test/lanelet2_python_api_for_autoware/setup.bash
python3 main.py --cache_file /home/ukenryu/map_test/lanelet2_python_api_for_autoware/extract_bags/caches/cache.json \
    --map_file /home/ukenryu/autoware_map/odaiba_stable/lanelet2_map.osm \
    --epochs 90000  --lr 1e-4