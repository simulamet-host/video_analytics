import os
import subprocess
import streamlit as st
from video_preprocessing_app import VideoPreprocessing


class ImagePreprocessing:
    """
    This class is responsible for image preprocessing
    """

    def __init__(self, starting_path):
        self.starting_path = starting_path
        self.output_folder = None
        self.resize = None
        self.img_width = None
        self.img_height = None
        self.cmd_list = None

    def run(self):
        """
        This function runs the image preprocessing app.
        """
        st.write("Select a folder containing images inside the data directory.")
        video_preprocessing = VideoPreprocessing(self.starting_path, self.output_folder)
        self.images_path = video_preprocessing.folder_selector_ui(
            self.starting_path, button_id=1
        )
        self.output_folder = os.path.join(self.starting_path, ".npz")
        self.cmd_list = [
            "python",
            "../image_preprocessing.py",
            "--dir",
            self.images_path,
            "--output",
            self.output_folder,
        ]
        self.resize = st.checkbox("Resize images")
        if self.resize:
            self.img_width = st.number_input("Image width", value=224)
            self.img_height = st.number_input("Image height", value=224)
        self.cmd_list.extend(
            [
                "--resize",
                str(self.resize),
                "--img_width",
                str(self.img_width),
                "--img_height",
                str(self.img_height),
            ]
        )
        if st.button("Run"):
            results = subprocess.run(self.cmd_list, capture_output=True, check=False)
            if results.returncode == 0:
                output = results.stdout.decode("utf-8")
                st.text_area("Output", value=output, height=200)
                st.write("Image preprocessing completed successfully.")
                st.balloons()
            else:
                st.error("Image preprocessing failed!")
                error = results.stderr.decode("utf-8")
                st.text_area("Error", value=error, height=200)
