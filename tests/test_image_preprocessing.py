import os
import subprocess
import argparse
import numpy as np
import pytest
import cv2
import e2evideo.image_preprocessing as e2e_img_pre

@pytest.fixture
def image_folder(tmpdir):
    # create temporary image folder
    tmpdir.mkdir('images')
    # define colors for test images
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # Loop over the colors and create an image for each color
    for i, color in enumerate(colors):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :, :] = color
        cv2.imwrite(str(tmpdir.join(f'images/image_{i}.jpg')), img)
    # add clearnup function 
    def cleanup():
        # remove the temporary folder
        tmpdir.remove()
    # return the temporary folder
    yield tmpdir.join('images')
    # call the cleanup function
    cleanup()

# Test calling the function from another module
def test_get_images_call(image_folder):
    # call the function to read images from the folder
    assert os.path.isdir(str(image_folder))
    
    opt_dict = {'dir': str(image_folder), 'img_format': '*.jpg', 'resize': False, 'img_width': 224, 'img_height': 224, 'gray_scale': False}
    opt_ = argparse.Namespace(**opt_dict)
    images = e2e_img_pre.get_images(opt_)
    # check that the correct number of images were read
    assert images.shape[0] == 3
    
    opt_.resize = True
    opt_.img_width = 32
    opt_.img_height = 32
    images_resized = e2e_img_pre.get_images(opt_)
    # check that the size of the resized images is correct
    assert images_resized.shape[1:] == (32, 32, 3)
    
    # First dim is the number of images, second and third are the image dimensions, last is the number of channels.
    opt_.gray_scale = True
    images_gray = e2e_img_pre.get_images(opt_)
    # check that the imgages are converted to grayscale (that means the last dimension is dropped, i.e. 4 -> 3)
    assert len(images_gray.shape) == 3

# Test running the module from the command line
def test_get_images_term(image_folder):
    script_dir = '../e2evideo/image_preprocessing.py'
    # check that the temporary folder exists
    assert os.path.isdir(str(image_folder))
    
    # call the python script to read images from the folder
    cmd = ['python', script_dir , '--dir', str(image_folder)]
    process_default = subprocess.run(cmd, capture_output=True)
    assert process_default.returncode == 0
    
    # check that the correct number of images were read and saved
    cmd.extend(['--img_format', '*.jpg'])
    process_jpg = subprocess.run(cmd, capture_output=True)
    assert process_jpg.returncode == 0
    assert process_jpg.stdout == b'Images saved in array of arrays with size (3, 200, 200, 3)\n'

    # check that the size of the resized images is correct
    cmd.extend(['--resize', 'True', '--img_width', '32', '--img_height' ,'32'])
    print(cmd)
    process_resized = subprocess.run(cmd, capture_output=True)
    assert process_resized.returncode == 0
    assert process_resized.stdout == b'Images saved in array of arrays with size (3, 32, 32, 3)\n'

    # check that the imgages are converted to grayscale (that means the last dimension is dropped, i.e. 4 -> 3)
    cmd.extend(['--gray_scale', 'True'])
    process_gray = subprocess.run(cmd, capture_output=True)
    assert process_gray.returncode == 0
    assert process_gray.stdout == b'Images saved in array of arrays with size (3, 32, 32)\n'
