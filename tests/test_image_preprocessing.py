import os
import numpy as np
import pytest
import cv2
from e2evideo import image_preprocessing as e2e_img_pre


@pytest.fixture
def image_folder(tmpdir):
    # create temporary image folder
    tmpdir.mkdir("images")
    # define colors for test images
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # Loop over the colors and create an image for each color
    for i, color in enumerate(colors):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :, :] = color
        cv2.imwrite(str(tmpdir.join(f"images/image_{i}.jpg")), img)

    # add clearnup function
    def cleanup():
        # remove the temporary folder
        tmpdir.remove()

    # return the temporary folder
    yield tmpdir.join("images")
    # call the cleanup function
    cleanup()


class TestImagesPreprocessor:
    """unit tess for image_preprocessing.py"""

    def test_get_all_images(self, image_folder):
        """Test that all images are read from the folder."""
        # call the function to read images from the folder
        assert os.path.isdir(image_folder)
        images_config = e2e_img_pre.ImagesConfig(
            dir=image_folder,
            img_format="*.jpg",
            gray_scale=False,
            output="all_images.npy",
            resize=False,
            img_width=224,
            img_height=224,
        )

        images_preprocessing = e2e_img_pre.ImagePreprocessing(images_config)
        images, _ = images_preprocessing.get_images()

        # check that the correct number of images were read
        assert images.shape[1] == 3, "Wrong number of images read"
        assert images.shape[2:] == (200, 200, 3), "Wrong image size"

    def test_get_all_images_resize(self, image_folder):
        """Test get gray scale option."""
        # call the function to read images from the folder

        images_config = e2e_img_pre.ImagesConfig(
            dir=image_folder,
            img_format="*.jpg",
            gray_scale=True,
            output="all_images.npy",
            resize=True,
            img_width=32,
            img_height=32,
        )

        images_preprocessing = e2e_img_pre.ImagePreprocessing(images_config)
        images_grayscale, _ = images_preprocessing.get_images()

        # check that the correct number of images were read
        assert images_grayscale.shape[1] == 3, "Wrong number of images read"
        assert images_grayscale.shape[2:] == (32, 32), "Wrong image size"
        assert len(images_grayscale.shape) == 4, "Wrong number of dimensions"

    def test_create_video(self, image_folder):
        """Test that the video is created."""
        # call the function to read images from the folder
        assert os.path.isdir(image_folder)
        images_config = e2e_img_pre.ImagesConfig(
            dir=image_folder,
            img_format="*.jpg",
            gray_scale=False,
            output="all_images.npy",
            resize=False,
            img_width=224,
            img_height=224,
        )

        images_preprocessing = e2e_img_pre.ImagePreprocessing(images_config)

        # create video with a given file name
        output_path = "test_video.avi"
        images_preprocessing.create_videos(str(image_folder), output_path, "jpg")
        # check that the video was created
        assert os.path.isfile(output_path), "Video not created"
        # check that the video has the correct size
        assert os.path.getsize(output_path) > 0, "Video is empty"
        # remove the video
        os.remove(output_path)

        # create video without given file name
        images_preprocessing.create_videos(str(image_folder), str(image_folder), "jpg")
        video_path = str(image_folder) + "new_video.avi"
        # check that the video was created
        assert os.path.isfile(video_path), "Video not created"
        # check that the video has the correct size
        assert os.path.getsize(video_path) > 0, "Video is empty"
        # remove the video
        os.remove(video_path)
