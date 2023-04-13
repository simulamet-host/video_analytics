#!/bin/bash
set -e

echo 'e2evideo pipeline'

echo 'Video preprocessing'
python video_preprocessing.py --videos_folder '../data/ucf_sports_actions/videos/' \
--images_folder '../data/ucf_sports_actions/frames/' \
--video_format 'avi'  --how_often 'per_second'  --output_folder '../data/ucf_sports_actions/frames'
echo 'Done'

echo 'Image preprocessing'
python image_preprocessing.py --dir '../data/ucf_sports_actions/frames' --resize 'True' \
--img_width '60' --img_height '60' --output './results/ucf_sports_actions_images.npz'
echo 'Done'

echo 'Feature Extractor'
python feature_extractor.py --images_array './results/ucf_sports_actions_images.npz' \
--data_folder '../data/ucf_sports_actions/frames' --no_classes '11' --mode 'train'
echo 'Done'

echo 'Downstreaming task :: action recognition'
python action_recogniton.py --images_array './results/ucf_sports_actions_images.npz' \
--data_folder '../data/ucf_sports_actions/frames' --no_classes '11'  --mode 'train'
