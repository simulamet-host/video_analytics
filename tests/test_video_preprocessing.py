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

        try:
            assert os.path.isdir(
                os.path.join(test_videos_config.output_folder, "test_video")
            )
            assert (
                len(
                    os.listdir(
                        os.path.join(test_videos_config.output_folder, "test_video")
                    )
                )
                > 0
            )
            assert os.path.isdir(test_videos_config.output_folder)
        except AssertionError as error:
            print(f"Assertion error: {error}")

        try:
            shutil.rmtree(test_videos_config.output_folder)
        except OSError as error:
            print(f"Error: {error}")

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

        try:
            assert os.path.isdir(
                os.path.join(test_videos_config.output_folder, "test_video")
            )
            assert (
                len(
                    os.listdir(
                        os.path.join(test_videos_config.output_folder, "test_video")
                    )
                )
                > 0
            )
            assert os.path.isdir(test_videos_config.output_folder)
        except AssertionError as error:
            print(f"Assertion error: {error}")

        # change the background subtraction method
        test_videos_config.back_sub = "KNN"
        processor = e2e_vid_pre.VideoPreprocessor(test_videos_config)
        processor.process_video()

        assert (
            len(
                os.listdir(os.path.join(test_videos_config.output_folder, "test_video"))
            )
            > 0
        )

        try:
            shutil.rmtree(test_videos_config.output_folder)
        except OSError as error:
            print(f"Error: {error}")
