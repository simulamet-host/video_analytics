import os
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
        img.save(tmpdir.join(f'images/image_{i}.jpg'))
    return tmpdir.join('images')

def test_get_images(image_folder):
    # call the function to read images from the folder
    assert os.path.isdir(str(image_folder))
    images = e2e_img_pre.get_images(str(image_folder), '*.jpg', re_size=False, gray_scale=False)

    # check that the correct number of images were read
    print(images.shape)
    assert images.shape[0] == 3

    # check that the image data is correct
    for i in range(3):
        with open(str(image_folder.join(f'image_{i}.jpg')), 'rb') as f:
            assert images[i].tobytes() == f.read()

if __name__ == '__main__':
    test_get_images(image_folder)
