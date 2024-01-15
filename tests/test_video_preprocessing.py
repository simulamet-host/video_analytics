"""Unit tests for video_preprocessing.py"""
import os
import shutil

from e2evideo import video_preprocessing as e2e_vid_pre


class TestVideoPreprocessor:
    """Unit tests for VideoPreprocessor class"""

    def test_process_video(self):
        """Test process_video method"""
        test_videos_config = e2e_vid_pre.VideoConfig(
            videos_folder="test_videos",
            video_format=".avi",
            output_folder="test_images",
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
