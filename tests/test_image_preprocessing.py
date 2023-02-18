import os
import subprocess
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
    return tmpdir.join('images')

def test_get_images(image_folder):
    # call the function to read images from the folder
    assert os.path.isdir(str(image_folder))
    images = e2e_img_pre.get_images(str(image_folder), '*.jpg', re_size=False, gray_scale=False)
    # check that the correct number of images were read
    print(images.shape)
    assert images.shape[0] == 3

    images_resized = e2e_img_pre.get_images(str(image_folder), '*.jpg', re_size=True, gray_scale=False)
    # check that the size of the resized images is correct
    assert images_resized.shape[1:] == (224, 224, 3)
    # First dim is the number of images, second and third are the image dimensions, last is the number of channels.
    images_gray = e2e_img_pre.get_images(str(image_folder), '*.jpg', re_size=False, gray_scale=True)
    # check that the imgages are converted to grayscale (that means the last dimension is dropped, i.e. 4 -> 3)
    assert len(images_gray.shape) == 3

def test_main(image_folder):
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
    resize_dim_str = f"(32,32)"  # Convert to string representation of tuple
    cmd.extend(['--is_resize', 'True', '--resize_dim', resize_dim_str])
    print(cmd)
    process_resized = subprocess.run(cmd, capture_output=True)
    assert process_resized.returncode == 0
    assert process_resized.stdout == b'Images saved in array of arrays with size (3, 32, 32, 3)\n'

if __name__ == '__main__':
    #test_get_images(image_folder)
    test_main(image_folder)
