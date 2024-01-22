"""Unit tests for video_preprocessing.py"""
import os
import shutil
import argparse
import pytest
from unittest.mock import patch
from e2evideo import video_preprocessing as e2e_vid_pre

OUTPUT_FOLDER = "test_images"


@pytest.fixture
def mock_args():
    with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
        mock_parse_args.return_value = argparse.Namespace(
            videos_folder="test_videos",
            video_format=".avi",
            image_format="jpg",
            sampling_mode="every_frame",
            num_frames=10,
            output_folder=OUTPUT_FOLDER,
            backsub=None,
            save_frames=False,
        )
        yield


def clean_up_folder(folder):
    try:
        shutil.rmtree(folder)
    except OSError as error:
        print(f"Error: {error}")


def assert_output(out_folder):
    try:
        assert os.path.isdir(os.path.join(out_folder, "test_video"))
        assert len(os.listdir(os.path.join(out_folder, "test_video"))) > 0
        assert os.path.isdir(out_folder)

    except AssertionError as error:
        print(f"Assertion error: {error}")


def test_main_function(mock_args):
    e2e_vid_pre.main()
    try:
        assert os.path.isdir(os.path.join(OUTPUT_FOLDER, "test_video"))
        assert len(os.listdir(os.path.join(OUTPUT_FOLDER, "test_video"))) > 0
        assert os.path.isdir(OUTPUT_FOLDER)
    except AssertionError as error:
        print(f"Assertion error: {error}")
    try:
        shutil.rmtree(OUTPUT_FOLDER)
    except OSError as error:
        print(f"Error: {error}")


class TestVideoPreprocessor:
    """Unit tests for VideoPreprocessor class"""

    def test_process_video(self):
        """Test process_video method"""
        test_videos_config = e2e_vid_pre.VideoConfig(
            videos_folder="test_videos",
            video_format=".avi",
            output_folder=OUTPUT_FOLDER,
        )
        processor = e2e_vid_pre.VideoPreprocessor(test_videos_config)
        processor.process_video()

        assert_output(test_videos_config.output_folder)
        # try to extract frames again
        processor.process_video()
        assert_output(test_videos_config.output_folder)
        clean_up_folder(test_videos_config.output_folder)

        # change sampling method
        sampling_mode_list = ["fixed_frames", "every_frame", "per_second", "per_minute"]
        for sampling_mode in sampling_mode_list:
            test_videos_config.sampling_mode = sampling_mode
            processor = e2e_vid_pre.VideoPreprocessor(test_videos_config)
            processor.process_video()
            assert_output(test_videos_config.output_folder)
            clean_up_folder(test_videos_config.output_folder)

        # fixed-frames mode with higher number of frames
        test_videos_config.sampling_mode = "fixed_frames"
        test_videos_config.num_frames = 500
        processor = e2e_vid_pre.VideoPreprocessor(test_videos_config)
        processor.process_video()
        assert_output(test_videos_config.output_folder)
        clean_up_folder(test_videos_config.output_folder)

        # save frames
        test_videos_config.save_frames = True
        processor = e2e_vid_pre.VideoPreprocessor(test_videos_config)
        processor.process_video()
        assert_output(test_videos_config.output_folder)
        clean_up_folder(test_videos_config.output_folder)

    def test_process_video_with_backsub(self):
        """Test process_video method with background subtraction"""
        test_videos_config = e2e_vid_pre.VideoConfig(
            videos_folder="test_videos",
            video_format=".avi",
            output_folder=OUTPUT_FOLDER,
            back_sub="MOG2",
        )
        processor = e2e_vid_pre.VideoPreprocessor(test_videos_config)
        processor.process_video()

        assert_output(test_videos_config.output_folder)
        clean_up_folder(test_videos_config.output_folder)

        # change the background subtraction method
        test_videos_config.back_sub = "KNN"
        processor = e2e_vid_pre.VideoPreprocessor(test_videos_config)
        processor.process_video()

        assert_output(test_videos_config.output_folder)
        clean_up_folder(test_videos_config.output_folder)
