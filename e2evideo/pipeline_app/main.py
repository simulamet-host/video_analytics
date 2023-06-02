"""
This module contains the streamlit app for the e2eVideo pipeline.
"""
import os
import streamlit as st
from PIL import Image
from utils import colored_text
from video_preprocessing_app import VideoPreprocessing
from image_preprocessing_app import ImagePreprocessing

class StreamlitApp:
    """
    This class contains the streamlit app for the e2eVideo pipeline.
    """
    def __init__(self):
        self.img = Image.open("logo.png")
        self.starting_path = self.set_starting_path()
        self.output_folder = os.path.join(self.starting_path, 'frames')

    def set_starting_path(self):
        """
        This function sets the starting path for the folder selector.
        Parameters
        ----------
        None
        Returns
        -------
            starting_path : str
                The path to the starting directory.
        """
        # Set the starting path to the 'data' directory
        current_directory = os.getcwd()
        parent_directory_ = os.path.dirname(current_directory)
        parent_directory = os.path.dirname(parent_directory_)
        starting_path = os.path.join(parent_directory, 'data')
        # Check if the data directory exists
        if not os.path.isdir(starting_path):
            os.mkdir(starting_path)
            # change the following line to a colored text
            colored_text("New data directory is created!", "green")
        return starting_path

    def run(self):
        """
        This function runs the streamlit app.
        """
        st.image(self.img, width=500)
        st.title('e2eVideo Pipeline App')
        colored_text("Please place your data files in a folder inside the 'data' folder,\
              located within the same directory as the running project.", "Salmon")

        with st.expander("Video Preprocessing"):
            video_preprocessing = VideoPreprocessing(self.starting_path, self.output_folder)
            video_preprocessing.get_frames()

        with st.expander("Image Preprocessing"):
            image_preprocessing = ImagePreprocessing(self.output_folder)
            image_preprocessing.run()
    
        with st.expander("Feature Extraction"):
            st.write('Select a folder containing images inside the data directory.')

        with st.expander("Classification"):
            st.write('Select a folder containing images inside the data directory.')

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
