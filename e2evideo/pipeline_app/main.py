"""
This module contains the streamlit app for the e2eVideo pipeline.
"""
import os
import glob
import random
import subprocess
import streamlit as st
from PIL import Image
from utils import colored_text, get_subdirectories, get_parent_directory

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

    def folder_selector_ui(self, input_path):
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

        st.write('Data directory: ', input_path)

        selected_dir = st.selectbox('Select video folder', st.session_state.subdirs,
                                    key="video_folder_selectbox")
        column =  st.columns(3)

        if column[0].button('Select folder', key="select_folder_button"):
            new_path = os.path.join(st.session_state.selected_path, selected_dir)
            # if it is a correct folder, update the selected path
            if os.path.isdir(new_path):
                st.session_state.selected_path = new_path
            else:
                return st.session_state.selected_path
            if get_subdirectories(new_path):
                st.session_state.subdirs = get_subdirectories(new_path)
                # show button to show subdirectories
                if column[1].button('Open Folder', key="show_subdirs_button"):
                    st.session_state.selected_path = new_path
                    st.session_state.subdirs = get_subdirectories(new_path)
                st.write('Selected folder:', st.session_state.selected_path)
            else:
                colored_text("There are no subdirectories in this folder.", "gray")
                st.write('Selected folder:', st.session_state.selected_path)

        if column[2].button('Go up', key = "go_up_button"):
            parent_dir = get_parent_directory(st.session_state.selected_path)
            st.session_state.selected_path = parent_dir[0]
            st.session_state.subdirs = get_subdirectories(parent_dir[0])
        return st.session_state.selected_path

    def run(self):
        """
        This function runs the streamlit app.
        """
        st.image(self.img, width=500)
        st.title('e2eVideo Pipeline App')
        colored_text("Please place your data files in a folder inside the 'data' folder,\
              located within the same directory as the running project.", "Salmon")

        with st.expander("Video Preprocessing"):
            selected_folder = self.folder_selector_ui(self.starting_path)
            cmd_list = ['python', '../video_preprocessing.py', '--videos_folder', selected_folder]
            # check if output folder exists and create it if it doesn't
            if not os.path.isdir(self.output_folder):
                os.mkdir(self.output_folder)
                colored_text("New frames directory is created!", "green")
            colored_text("Frames will be saved in the following directory:", "gray")
            st.write(self.output_folder)

            cmd_list.extend(['--output_folder', self.output_folder])

            video_format = st.selectbox('Select video format', ('mp4', 'avi', 'mov', 'wmv', 'flv',
                                                                'mkv', 'webm', 'm4v', '3gp'),
                                        key='video_format_selectbox')
            image_format = st.selectbox('Select image format', ('jpg', 'png', 'bmp', 'tiff',
                                        'gif', 'webp', 'ico', 'raw', 'eps', 'psd', 'svg'),
                                        key='image_format_selectbox')
            sampling_mode = st.selectbox('Select sampling mode', ('every_frame', 'per_second',
                                                                'fixed_frames'),
                                                                key='sampling_mode_selectbox')
            cmd_list.extend(['--video_format', video_format, '--image_format', image_format,
                            '--sampling_mode', sampling_mode])

            if sampling_mode == 'fixed_frames':
                num_frames = st.number_input('Number of frames', min_value=1, max_value=1000,
                                            value=10, step=1)
                cmd_list.extend(['--num_frames', str(num_frames)])

            if "extract_frames_clicked" not in st.session_state:
                st.session_state.extract_frames_clicked = False

            if st.button('Extract Frames', key='extract_frames_button'):
                st.session_state.extract_frames_clicked = True
                results = subprocess.run(cmd_list, capture_output=True, check=False)
                if results.returncode == 0:
                    output = results.stdout.decode('utf-8')
                    st.text_area("Output", value=output, height=200)
                    st.success('Frames extracted successfully!')
                else:
                    st.error('Error extracting frames!')
                    error = results.stderr.decode('utf-8')
                    st.write(f'Error: \n {error}')

            if st.session_state.extract_frames_clicked:
                if st.button('Show Example Frames', key='show_example_button'):
                    frames_subdir = get_subdirectories(self.output_folder)
                    random_frames_subdir = random.sample(frames_subdir, k=3)
                    for subdir in random_frames_subdir:
                        frames = glob.glob(os.path.join(self.output_folder,
                                                        subdir, "*." + image_format))
                        frames.sort()
                        selected_frames = frames[:3]
                        st.image(selected_frames, width=500)

        with st.expander("Image Preprocessing"):
            st.write('Select a folder containing images inside the data directory.')

        with st.expander("Feature Extraction"):
            st.write('Select a folder containing images inside the data directory.')

        with st.expander("Classification"):
            st.write('Select a folder containing images inside the data directory.')

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
