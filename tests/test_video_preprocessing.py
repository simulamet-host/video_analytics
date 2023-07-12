"""Unit tests for video_preprocessing.py"""
import itertools
import unittest
from unittest.mock import Mock, patch, call
from e2evideo.video_preprocessing import VideoConfig, VideoPreprocessor


class StatefulMock(Mock):
    """Mock class that keeps track of the number of times it was called"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0

    def isOpened(self): # pylint: disable=invalid-name
        """Mock isOpened method"""
        self.call_count += 1
        if self.call_count > 10:  # Adjust this value to fit your test case
            return False
        return True

class TestVideoPreprocessor(unittest.TestCase):
    """Unit tests for VideoPreprocessor class"""
    @patch('os.makedirs')
    def test_create_folder(self, mock_makedirs):
        """Test create_folder method"""
        config = VideoConfig(videos_folder='test_videos')
        processor = VideoPreprocessor(config)
        processor.create_folder('test_folder')

        mock_makedirs.assert_called_once_with('test_folder')

    @patch('os.makedirs')
    @patch('os.path.isdir')
    @patch('glob.glob')
    def test_get_video_files(self, mock_glob, mock_isdir, mock_makedirs):
        """Test get_video_files method"""
        mock_isdir.return_value = True
        mock_glob.return_value = ['test_video.mp4']

        config = VideoConfig(videos_folder='test_videos')
        processor = VideoPreprocessor(config)
        video_files = processor.get_video_files()

        mock_glob.assert_called_once_with('test_videos/**/*mp4', recursive=True)
        self.assertEqual(video_files, ['test_video.mp4'])

    @patch('cv2.VideoCapture')
    @patch('e2evideo.video_preprocessing.VideoPreprocessor.get_video_files')
    def test_process_video(self, mock_get_video_files, mock_cv2_capture):
        """Test process_video method"""
        # Mock get_video_files to return a dummy video file
        mock_get_video_files.return_value = ['test_video.mp4']
        mock_cv2_capture_instance = StatefulMock()
        mock_cv2_capture_instance.read.return_value = (True, 'test_frame')
        mock_cv2_capture_instance.get.side_effect = itertools.chain([24.0, 10.0, 1.0, False],
                                                                    itertools.repeat(False))
        mock_cv2_capture.return_value = mock_cv2_capture_instance

        config = VideoConfig(videos_folder='test_videos', sampling_mode='every_frame',
                             output_folder = 'test_videos/frames/', save_frames=True)
        processor = VideoPreprocessor(config)

        with patch('cv2.imwrite') as mock_imwrite, patch('os.makedirs') as _:
            frames = processor.process_video()

        # Verify if frames were processed correctly
        self.assertEqual(len(frames), 10)
        self.assertTrue(all(frame == 'test_frame' for frame in frames))

        # Verify if frames were saved correctly
        calls = [call(f'test_videos/frames/test_video/frame{index}.jpg',
                      'test_frame') for index in range(10)]
        mock_imwrite.assert_has_calls(calls, any_order=True)

if __name__ == '__main__':
    unittest.main()
