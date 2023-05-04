import os
import glob
import subprocess
import streamlit as st

st.title('e2e-Video')

with st.sidebar:
    st.write('St Session State' , st.session_state)

st.write(""" Please place your data files in a folder inside the 'data',
located within the same directory as the running project.""")

def get_subdirectories(path):
    subdirs = [dir_name for dir_name in os.listdir(path) if os.path.isdir(os.path.join(path, dir_name))]
    return subdirs

def get_parent_directory(path):
    parent_dir = [os.path.dirname(path)]
    return parent_dir

def folder_selector_ui(starting_path):
    
    if "subdirs" not in st.session_state:
        st.session_state.subdirs = get_subdirectories(starting_path)
    if "start_button_clicked" not in st.session_state:
        st.session_state.start_button_clicked = False
    if "selected_path" not in st.session_state:
        st.session_state.selected_path = starting_path
    
    st.write('Data directory: ', starting_path)
    st.write('Select a folder containing videos inside the data directory.')
    
    # if subdirs not empty create a button to show subdirs
    selected_dir = st.selectbox('Select video folder', st.session_state.subdirs, key="video_folder_selectbox")
    column =  st.columns(3)
    if get_subdirectories(os.path.join(st.session_state.selected_path, st.session_state.video_folder_selectbox)) != []:
        st.session_state.selected_path = os.path.join(st.session_state.selected_path, selected_dir)
        st.session_state.subdirs = get_subdirectories(st.session_state.selected_path)
        
        if column[0].button('Open', key="show_subfolders_button"):
            st.session_state.subdirs = get_subdirectories(st.session_state.selected_path)
    if column[1].button('Go up', key = "go_up_button"):
        st.session_state.subdirs = get_parent_directory(st.session_state.selected_path)
    #create a button to select the folder
    if column[2].button('Select folder', key="select_folder_button"):
        st.session_state.start_button_clicked = True
        selected_path = os.path.join(starting_path, selected_dir)
        st.write('Selected folder:', selected_path)
        return selected_path
    else:
        return starting_path
    

# Set the starting path to the 'data' directory
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
starting_path = os.path.join(parent_directory, 'data')

# Check if the data directory exists
if not os.path.isdir(starting_path):
    os.mkdir(starting_path)
    st.write('Created data directory!')
else:
    st.write('Data directory exists!')

selected_folder = folder_selector_ui(starting_path)

if st.session_state.start_button_clicked:
    cmd_list = ['python', 'video_preprocessing.py', '--videos_folder', selected_folder]
    with st.echo():
        output_folder = os.path.join(starting_path, 'frames')
    cmd_list.extend(['--output_folder', output_folder])

    video_format = st.selectbox('Select video format', ('mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv', 'webm', 'm4v', '3gp'),
                                key='video_format_selectbox')
    image_format = st.selectbox('Select image format', ('jpg', 'png', 'bmp', 'tiff', 'gif', 'webp', 'ico', 'raw', 'eps', 'psd', 'svg'),
                                key='image_format_selectbox')
    sampling_mode = st.selectbox('Select sampling mode', ('every_frame', 'per_second', 'fixed_frames'),
                                    key='sampling_mode_selectbox')
    cmd_list.extend(['--video_format', video_format, '--image_format', image_format, '--sampling_mode', sampling_mode])

    if sampling_mode == 'fixed_frames':
        num_frames = st.number_input('Number of frames', min_value=1, max_value=1000, value=10, step=1)
        cmd_list.extend(['--num_frames', str(num_frames)])

    if st.button('Extract Frames', key='extract_frames_button'):
        results = subprocess.run(cmd_list, capture_output=True)
        if results.returncode == 0:
            output = results.stdout.decode('utf-8')
            st.write(f'Output: \n {output}')
            st.success('Frames extracted successfully!')
        else:
            st.error('Error extracting frames!')
            error = results.stderr.decode('utf-8')
            st.write(f'Error: \n {error}')
