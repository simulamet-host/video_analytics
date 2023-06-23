"""
This model is used to predict the action in a video.
"""
#%%
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from mxnet import nd
from gluoncv.model_zoo import get_model
import cv2
from video_preprocessing import VideoConfig, VideoPreprocessor
from image_preprocessing import get_images

def select_frames(video_frames):
    total_frames = len(video_frames)
    step = total_frames // 5  # Determine step size
    selected_frames = video_frames[::step]  # Select frames
    return selected_frames[:5]  # Get the first 5 frames


def plot_predictions(video_frames, net):
    """
    This function is used to plot the predictions of the model.
    """
    label_data = pd.read_csv("../datasets/ucf101/ucfTrainTestlist/classInd.txt", sep=' ', header=None,
                             engine="pyarrow")
    label_data.columns=['index', 'labels']

    counter = 1
    figure_counter = 1
    

    for video in video_frames:
        plt.figure(figsize=(3, 3))
        selected_frames = select_frames(video)
        final_pred = 0
        for _, frame_img in enumerate(selected_frames):
            frame_array = nd.array(frame_img).expand_dims(axis=0).transpose((0, 3, 1, 2))
            pred = net(frame_array)
            final_pred += pred
        final_pred /= len(selected_frames)

        classes = net.classes
        top_classes = 5
        ind = nd.topk(final_pred, k=top_classes)[0].astype('int')
        print('\n \n ----------- \n \n')
        print('The input video is classified to be')
        for i in range(top_classes):
            top_ = classes[ind[i].asscalar()]
            prob_ = nd.softmax(final_pred)[0][ind[i]].asscalar()
            print(f'\t{top_}, with probability {prob_}.')

        # Select a frame to plot
        frame = selected_frames[0, :, :, :]
        frame_resize = cv2.resize(frame, (60, 60)) #pylint: disable=no-member
        plt.imshow(frame_resize)
        plt.title(classes[ind[0].asscalar()], fontstyle='italic', backgroundcolor='pink', pad=10)
        counter += 1
        plt.axis('off')
        plt.show()
        counter += 1
        figure_counter += 1
    plt.show()
    print('\n \n')
    print('Done')

#%%
net_ = get_model('vgg16_ucf101', nclass=101, pretrained=True)
# check if folder ..data is not created then create it
if not os.path.exists('../datasets/ucf101/'):
    os.makedirs('../datasets/ucf101/')

videos_config = VideoConfig('../datasets/ucf101/', 'mp4', 'jpg',
                            'every_frame', 10, '../datasets/ucf101/frames', None)
processor = VideoPreprocessor(videos_config)
frames_data = processor.process_video()
#%%
opt_dict = {'dir': '../datasets/ucf101/frames', 'img_format': '*.jpg',
            'resize': True, 'img_width': 224, 'img_height': 224, 'gray_scale': False, 
            'output': 'all_images.npy'}
opt_ = argparse.Namespace(**opt_dict)
frames_, _ , = get_images(opt_)
plot_predictions(frames_, net_)
# %%
