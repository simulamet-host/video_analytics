"""
Video pre-processing module that extract frames from videos.
Main functionality:
    * Create './images' folder, if it does not exist already.
    * Read all videos from a given videos folder and a specified video extension
    (default mp4).
    * Choose between either Saving all frames or one frame per second,
        in a video in corresponding images folder with the same video file name.
"""
# pylint: disable=no-member
# import packages needed
import argparse
from dataclasses import dataclass
import os
import glob
import math
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
OUTPUT_PATH = "../data/ucf_sports_actions/frames"


@dataclass
class VideoConfig:
    """Configuration class for video pre-processing."""

    videos_folder: str
    video_format: str = "mp4"
    image_format: str = "jpg"
    sampling_mode: str = "every_frame"
    num_frames: int = 10
    output_folder: str = OUTPUT_PATH
    back_sub: str = None
    save_frames: bool = True


@dataclass
class FrameData:
    """Data class for frame data."""

    frame: any = None
    frames: list = None
    frame_indices: list = None
    count: int = 0
    frames_folder: str = None
    frame_rate: int = 0


back_sub_algo_mapping = {
    "MOG2": cv2.createBackgroundSubtractorMOG2(),  # pylint: disable=E1101
    "KNN": cv2.createBackgroundSubtractorKNN(),  # pylint: disable=E1101
}


class VideoPreprocessor:
    """Video pre-processing class that extract frames from videos."""

    def __init__(self, config: VideoConfig):
        """initialize the class."""
        self.config = config

    def create_folder(self, dir_):
        """
        create a folder at dir if it does not exist already.
        Input:
            - dir: the folder directory
        """
        try:
            os.makedirs(dir_)
        except OSError:
            pass

    def get_video_files(self):
        """Get all video files from the given videos folder."""
        assert os.path.isdir(
            self.config.videos_folder
        ), "The given videos_folder does not exist"
        self.create_folder(self.config.output_folder)
        search_path = os.path.join(
            self.config.videos_folder, "**", "*" + self.config.video_format
        )
        video_files = glob.glob(search_path, recursive=True)
        assert (
            len(video_files) != 0
        ), "The given videos folder does not contain any vidoes"
        return video_files

    def calculate_frame_indices(self, frame_rate, total_num_frames, video_file):
        """Calculate the frame indices for the fixed-frames mode."""
        if self.config.num_frames > total_num_frames:
            print(
                f"Warning: the number of fixed-frames is larger than the total \
                number of frames"
                f"in the video {video_file}"
            )
            frame_indices = list(range(int(total_num_frames)))
        else:
            frame_indices = [
                round(frame_rate * i) for i in range(self.config.num_frames)
            ]
        return frame_indices

    def interpolate_missing_frames(self, frame_data: FrameData):
        """Interpolate missing frames if the number of frames is less than the number \
        of frames."""
        while len(frame_data.frames) < self.config.num_frames:
            alpha = float(len(frame_data.frames)) / len(frame_data.frame_indices)
            # pylint: disable=E1101
            interpolated_frame = cv2.addWeighted(
                frame_data.frame, 1 - alpha, frame_data.frames[-1], alpha, 0
            )
            frame_data.frames.append(interpolated_frame)
            file_name = f"frame{frame_data.count}." + self.config.image_format
            cv2.imwrite(frame_data.frames_folder + "/" + file_name, interpolated_frame)
            frame_data.count += 1
        return frame_data.count

    def process_video(self):
        """
        Main function will run when the python file is called from terminal.
        """
        video_files = self.get_video_files()
        for video_file in video_files:
            print("\n\n")
            frame_data = FrameData()
            # video_format_len = len(self.config.video_format) + 1
            images_sub_folder = video_file.split("/")[-1].split(".")[0]
            frame_data.frames_folder = os.path.join(
                self.config.output_folder, images_sub_folder
            )
            self.create_folder(frame_data.frames_folder)

            # if the frames are already extracted, remove them
            if len(os.listdir(frame_data.frames_folder)) != 0:
                print(
                    f"Warning: the frames folder {frame_data.frames_folder} \
                    is not empty"
                )
                print("Removing existing frames...")
                for file_name in os.listdir(frame_data.frames_folder):
                    os.remove(os.path.join(frame_data.frames_folder, file_name))

            print(f"Extracting frames from {video_file}...")
            # capture the video from the video file
            cap = cv2.VideoCapture(video_file)  # pylint: disable=E1101
            frame_data.frame_rate = cap.get(cv2.CAP_PROP_FPS)  # pylint: disable=E1101
            total_num_frames = cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )  # pylint: disable=E1101
            frame_data.frame_indices = self.calculate_frame_indices(
                frame_data.frame_rate, total_num_frames, video_file
            )
            frame_data.count = 0
            frame_data.frames = []
            cond1 = self.config.sampling_mode == "fixed_frames"
            if self.config.back_sub in back_sub_algo_mapping:
                back_sub_algo = back_sub_algo_mapping[self.config.back_sub]
            else:
                print(
                    f"No background subtraction algorithm \
                    provided : {self.config.back_sub}"
                )
            while cap.isOpened():
                frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)  # pylint: disable=E1101
                ret, frame_data.frame = cap.read()
                if not ret:
                    break
                file_name = f"frame{frame_data.count}." + self.config.image_format
                masked_frame_name = None
                if self.config.back_sub is not None:
                    masked_frame_name = (
                        f"fgMask_frame{frame_data.count}." + self.config.image_format
                    )
                    fg_mask = back_sub_algo.apply(frame_data.frame)
                should_save_frame = False
                if self.config.sampling_mode == "every_frame":
                    should_save_frame = True
                elif self.config.sampling_mode == "per_second":
                    should_save_frame = (
                        frame_id % math.floor(frame_data.frame_rate) == 0
                    )
                elif self.config.sampling_mode == "per_minute":
                    frames_per_minute = math.floor(frame_data.frame_rate) * 60
                    should_save_frame = frame_id % frames_per_minute == 0
                elif cond1:
                    should_save_frame = frame_id in frame_data.frame_indices
                if should_save_frame:
                    cv2.imwrite(
                        frame_data.frames_folder + "/" + file_name, frame_data.frame
                    )  # pylint: disable=E1101
                    if masked_frame_name is not None:
                        # Apply the mask to the original frame
                        # pylint: disable=E1101
                        fg_frame = cv2.bitwise_and(
                            frame_data.frame, frame_data.frame, mask=fg_mask
                        )
                        cv2.imwrite(
                            frame_data.frames_folder + "/" + masked_frame_name, fg_frame
                        )
                    frame_data.count += 1
                    frame_data.frames.append(frame_data.frame)
                # interpolate missing frames if count is less than the number of frames
                cond2 = frame_id == total_num_frames - 1
                cond3 = len(frame_data.frames) < self.config.num_frames
                if cond1 and cond2 and cond3:
                    print(
                        (
                            f"Warning: the number of frames extracted from \
                                {video_file} is"
                            "less than the number of frames specified"
                        )
                    )
                    print("Interpolating missing frames...")
                    frame_data.count = self.interpolate_missing_frames(frame_data)
            cap.release()
            if self.config.save_frames:
                print(
                    f"Done! {frame_data.count} images of format JPG is "
                    f"saved in {frame_data.frames_folder}"
                )
            else:
                print("Done!")
            return frame_data.frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos_folder", required=True, help="Path to the videos folder."
    )
    parser.add_argument(
        "--video_format", default="mp4", help="choose the video format to read."
    )
    parser.add_argument(
        "--image_format", default="jpg", help="choose the format for the output images."
    )
    parser.add_argument(
        "--sampling_mode",
        default="every_frame",
        choices=["fixed_frames", "every_frame", "per_second", "per_minute"],
    )
    parser.add_argument("--num_frames", default=10, type=int)
    parser.add_argument("--output_folder", default=OUTPUT_PATH)
    parser.add_argument("--backsub", default=None, choices=["MOG2", "KNN"])
    parser.add_argument(
        "--save_frames", default="False", type=bool, help="Save the frames locally."
    )
    opts = parser.parse_args()
    videos_config = VideoConfig(
        opts.videos_folder,
        opts.video_format,
        opts.image_format,
        opts.sampling_mode,
        opts.num_frames,
        opts.output_folder,
        opts.backsub,
        opts.save_frames,
    )
    processor = VideoPreprocessor(videos_config)
    processor.process_video()


if __name__ == "__main__":
    main()
