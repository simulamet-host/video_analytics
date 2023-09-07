"""
This module contains the class for video preprocessing app for the e2eVideo pipeline.
"""
import os
import glob
import random
import subprocess
import streamlit as st
from utils import colored_text, get_subdirectories, get_parent_directory


class VideoPreprocessing:
    """
    This class contains the video preprocessing app for the e2eVideo pipeline.
    """

    def __init__(self, starting_path, output_folder):
        self.paths = {
            "starting_path": starting_path,
            "output_folder": output_folder,
            "selected_folder": None,
        }
        self.options = {
            "video_format": None,
            "image_format": None,
            "sampling_mode": None,
            "num_frames": None,
        }
        self.cmd_list = None

    def folder_selector_ui(self, input_path, button_id=0):
        """
        This function displays a folder selector user interface.
        Parameters
        ----------
        input_path : str
            The path to the starting directory.
        Returns
        -------
        selected_path : str
            The path to the selected directory.
        """
        if "subdirs" not in st.session_state:
            st.session_state.subdirs = get_subdirectories(input_path)
            st.session_state.selected_path = input_path

        st.write("Data directory: ", input_path)

        selected_dir = st.selectbox(
            "Select video folder",
            st.session_state.subdirs,
            key=f"video_folder_selector_{button_id}",
        )
        column = st.columns(3)

        if column[0].button("Select folder", key=f"select_folder_button_{button_id}"):
            new_path = os.path.join(st.session_state.selected_path, selected_dir)
            # if it is a correct folder, update the selected path
            if os.path.isdir(new_path):
                st.session_state.selected_path = new_path
            else:
                return st.session_state.selected_path
            if get_subdirectories(new_path):
                st.session_state.subdirs = get_subdirectories(new_path)
                # show button to show subdirectories
                if column[1].button(
                    "Open Folder", key=f"show_subdirs_button_{button_id}"
                ):
                    st.session_state.selected_path = new_path
                    st.session_state.subdirs = get_subdirectories(new_path)
                st.write("Selected folder:", st.session_state.selected_path)
            else:
                colored_text("There are no subdirectories in this folder.", "gray")
                st.write("Selected folder:", st.session_state.selected_path)

        if column[2].button("Go up", key=f"go_up_button_{button_id}"):
            parent_dir = get_parent_directory(st.session_state.selected_path)
            st.session_state.selected_path = parent_dir[0]
            st.session_state.subdirs = get_subdirectories(parent_dir[0])
        return st.session_state.selected_path

    def get_frames(self):
        """
        A fuction to get frames from videos.
        """
        self.paths["selected_folder"] = self.folder_selector_ui(
            self.paths["starting_path"]
        )
        self.cmd_list = [
            "python",
            "../video_preprocessing.py",
            "--videos_folder",
            self.paths["selected_folder"],
        ]

        if not os.path.isdir(self.paths["output_folder"]):
            os.mkdir(self.paths["output_folder"])
            colored_text("New frames directory is created!", "green")
        colored_text("Frames will be saved in the following directory:", "gray")
        st.write(self.paths["output_folder"])

        self.cmd_list.extend(["--output_folder", self.paths["output_folder"]])

        self.options["video_format"] = st.selectbox(
            "Select video format",
            ("mp4", "avi", "mov", "wmv", "flv", "mkv", "webm", "m4v", "3gp"),
            key="video_format_selectbox",
        )
        self.options["image_format"] = st.selectbox(
            "Select image format",
            (
                "jpg",
                "png",
                "bmp",
                "tiff",
                "gif",
                "webp",
                "ico",
                "raw",
                "eps",
                "psd",
                "svg",
            ),
            key="image_format_selectbox",
        )
        self.options["sampling_mode"] = st.selectbox(
            "Select sampling mode",
            ("every_frame", "per_second", "fixed_frames"),
            key="sampling_mode_selectbox",
        )
        self.cmd_list.extend(
            [
                "--video_format",
                self.options["video_format"],
                "--image_format",
                self.options["image_format"],
                "--sampling_mode",
                self.options["sampling_mode"],
            ]
        )

        if self.options["sampling_mode"] == "fixed_frames":
            self.options["num_frames"] = st.number_input(
                "Number of frames", min_value=1, max_value=1000, value=10, step=1
            )
            self.cmd_list.extend(["--num_frames", str(self.options["num_frames"])])

        if "extract_frames_clicked" not in st.session_state:
            st.session_state.extract_frames_clicked = False

        if st.button("Extract Frames", key="extract_frames_button"):
            st.session_state.extract_frames_clicked = True
            results = subprocess.run(self.cmd_list, capture_output=True, check=False)
            if results.returncode == 0:
                output = results.stdout.decode("utf-8")
                st.text_area("Output", value=output, height=200)
                st.success("Frames extracted successfully!")
            else:
                st.error("Error extracting frames!")
                error = results.stderr.decode("utf-8")
                st.write(f"Error: \n {error}")

        if st.session_state.extract_frames_clicked:
            if st.button("Show Example Frames", key="show_example_button"):
                frames_subdir = get_subdirectories(self.paths["output_folder"])
                random_frames_subdir = random.sample(frames_subdir, k=3)
                for subdir in random_frames_subdir:
                    frames = glob.glob(
                        os.path.join(
                            self.paths["output_folder"],
                            subdir,
                            "*." + self.options["image_format"],
                        )
                    )
                    frames.sort()
                    selected_frames = frames[:3]
                    st.image(selected_frames, width=500)
