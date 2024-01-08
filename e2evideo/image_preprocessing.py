"""
Image preprocessing module.
Main functionality:
    * Resize images according to a given input dimensions.
    * Convert images to grayscale.
"""
# pylint: disable=no-member
import os
from typing import Optional
import argparse
from dataclasses import dataclass
import glob
import numpy as np
import cv2
from skimage import img_as_float32


@dataclass
class ImagesConfig:
    """Class to hold the configuration of the images."""

    dir: str
    img_format: str
    resize: bool
    gray_scale: bool
    output: str
    img_width: Optional[int] = None
    img_height: Optional[int] = None


class ImagePreprocessing:
    """Images preprocessing class."""

    def __init__(self, config: ImagesConfig):
        self.config = config

    def get_images_helper(self, folder_name):
        """Helper function for get_images."""
        video_file = []
        images_ = glob.glob(folder_name + "/" + self.config.img_format)
        for img_ in images_:
            img = cv2.imread(img_)  # pylint: disable=E1101
            if self.config.gray_scale is True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # pylint: disable=E1101
            if self.config.resize is True:
                img = cv2.resize(
                    img, (self.config.img_width, self.config.img_height)
                )  # pylint: disable=E1101

            video_file.append(img_as_float32(img))
            continue
        return video_file

    def get_images(self):
        """
        Read images from a given dir, using specified image format.
        Additionally it allows for resizing the images.
        Parameters:
            - dir_: directory contains the images.
            - img_format: specify the images format, it is string in the format "*.xxx".
            - resize: this is a boolean, set True if resizing of the images are needed.
            - resize_dim: set to the required image dimensions.
                        It takes input in the format (height, width).
        Returns:
            -- all_images: an array of array, with values from the images in the
            dir.
            This array is of size = (num_images, height, width, no_channels)
        """
        img_folders = [x[0] for x in os.walk(self.config.dir)]
        all_videos = []
        labels = []
        for folder_name in img_folders:
            video_file = self.get_images_helper(folder_name)
            if len(video_file) != 0:
                # convert video_file to numpy array
                video_file = np.array(video_file)
                all_videos.append(video_file)
                labels.append(folder_name.split("/")[-1])

        assert (
            len(all_videos) != 0
        ), "The given images folder does not contain any frames"
        # find the maximum length of the videos (number of frames) in a video
        max_frames = max(len(x) for x in all_videos)
        # find the maximum shape of the arrays
        # max_shape = max([arr.shape for arr in all_images])
        # create a new array with the maximum shape
        # specify the desired shape of the padded arrays
        frame_dim = all_videos[0][0].shape
        frames_in_videos_dim = (len(all_videos), max_frames) + frame_dim
        frames_in_videos = np.zeros(frames_in_videos_dim, dtype=np.float64)

        # pad the shorter videos with zeros at the end to make them all the same length
        for index_, video_ in enumerate(all_videos):
            frames_in_videos[index_][0 : len(video_)] = video_
        # save frames_in_videos to a file
        np.savez_compressed(self.config.output, frames_in_videos)
        # save labels to frames_labels.txt file
        with open("frames_labels.txt", "w", encoding="utf-8") as my_file:
            for label in labels:
                my_file.write(label + "\n")
        return frames_in_videos, labels

    def create_videos(self, folder_path, output_path, image_format):
        """Create videos from the extracted frames."""
        img_array = []
        # check if output path has .avi extension
        if output_path[-4:] != ".avi":
            output_path = output_path + "new_video.avi"
        for filename in sorted(glob.glob(folder_path + "/*." + image_format)):
            img = cv2.imread(filename)  # pylint: disable=E1101
            height, width, _ = img.shape
            size = (width, height)
            img_array.append(img)
        # pylint: disable=E1101
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"DIVX"), 10, size)
        for _, img in enumerate(img_array):
            out.write(img)
        out.release()


def main():
    parser_ = argparse.ArgumentParser()
    parser_.add_argument("--dir", default="./images/")
    parser_.add_argument("--img_format", default="*.jpg")
    parser_.add_argument("--resize", default=False)
    parser_.add_argument("--img_width", default=224, type=int)
    parser_.add_argument("--img_height", default=224, type=int)
    parser_.add_argument("--gray_scale", default=False)
    parser_.add_argument("--output", default="./results/all_images.npz")

    args_ = parser_.parse_args()

    images_config = ImagesConfig(
        args_.dir,
        args_.img_format,
        args_.resize,
        args_.gray_scale,
        args_.output,
        args_.img_width,
        args_.img_height,
    )
    image_preprocessing = ImagePreprocessing(images_config)
    _images, _ = image_preprocessing.get_images()

    print("Images saved in array of array of size", str(_images.shape))


if __name__ == "__main__":
    main()
