import os
import subprocess
import streamlit as st
st.title('e2e-Video')

cmd_list = ['python', 'video_preprocessing.py']
with st.echo():
    videos_folder = '../data/ucf_sports_actions/videos/'
    output_folder = '../data/ucf_sports_actions/frames/'
cmd_list.extend(['--videos_folder', videos_folder, '--output_folder', output_folder])

video_format = st.selectbox('Select video format', ('mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv', 'webm', 'm4v', '3gp'))
image_format = st.selectbox('Select image format', ('jpg', 'png', 'bmp', 'tiff', 'gif', 'webp', 'ico', 'raw', 'eps', 'psd', 'svg'))
sampling_mode = st.selectbox('Select sampling mode', ('every_frame', 'per_second', 'fixed_frames'))
cmd_list.extend(['--video_format', video_format, '--image_format', image_format, '--sampling_mode', sampling_mode])

if sampling_mode == 'fixed_frames':
    num_frames = st.number_input('Number of frames', min_value=1, max_value=1000, value=10, step=1)
    cmd_list.extend(['--num_frames', str(num_frames)])
else:
    num_frames = None

if st.button('Extract Frames'):
    results = subprocess.run(cmd_list, capture_output=True)
    if results.returncode == 0:
        output = results.stdout.decode('utf-8')
        st.write(f'Output: \n {output}')
        st.success('Frames extracted successfully!')
    else:
        st.error('Error extracting frames!')
        error= results.stderr.decode('utf-8')
        st.write(f'Error: \n {error}')
