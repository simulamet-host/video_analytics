#!/bin/bash
set -e
# check if an argument is provided
if [ $# -eq 0 ]
  then
    echo "Please provide an input variable."
    exit 1
fi

input_variable="$1"

video_preprocessing() {
    echo 'e2evideo pipeline'
    echo 'Video preprocessing'
    python video_preprocessing.py --videos_folder '../data/ucf_sports_actions/videos/' \
    --images_folder '../data/ucf_sports_actions/frames/' \
    --video_format 'avi'  --sampling_mode 'fixed_frames' --num_frames 6 --output_folder '../data/ucf_sports_actions/frames'
    echo 'Done'
}

image_preprocessing(){
    echo 'Image preprocessing'
    python image_preprocessing.py --dir '../data/ucf_sports_actions/frames' --resize 'True' \
    --img_width '60' --img_height '60' --output './results/ucf_sports_actions_images.npz'
    echo 'Done'
}

feature_extractor(){
    echo 'Feature Extractor'
    python cae_feature_extractor.py --images_array './results/ucf_sports_actions_images.npz' \
    --data_folder '../data/ucf_sports_actions/frames/' --no_classes '11' --mode 'train' \
    --labels_file 'frames_labels.txt' --epochs 1000 --batch_size 5
    echo 'Done'
}

action_recognition(){
    echo 'Downstreaming task :: action recognition'
    python action_recognition.py --images_array './results/ucf_sports_actions_images.npz' \
    --data_folder '../data/ucf_sports_actions/frames/' --no_classes '11'  --mode 'train' \
    --labels_file 'frames_labels.txt'
    echo 'Done'
}

case $input_variable in
    "video_preprocessing")
        video_preprocessing
        ;;
    "image_preprocessing")
        image_preprocessing
        ;;
    "feature_extractor")
        feature_extractor
        ;;
    "action_recognition")
        action_recognition
        ;;
    "complete_pipeline")
        video_preprocessing
        image_preprocessing
        feature_extractor
        action_recognition
        ;;
    *)
        echo "Please provide a valid input variable."
        exit 1
        ;;
esac