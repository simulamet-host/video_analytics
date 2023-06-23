"""
This model is used to predict the action in a video.
"""
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import mxnet as mx
from mxnet import nd
from gluoncv.model_zoo import get_model
import cv2
from image_preprocessing import get_images

def plot_predictions(video_frames, net):
    """
    This function is used to plot the predictions of the model.
    """
    label_data = pd.read_csv("../data/UCF-101/ucfTrainTestlist/classInd.txt", sep=' ', header=None,
                             engine="pyarrow")
    label_data.columns=['index', 'labels']

    counter = 1
    figure_counter = 1

    for video in video_frames:
        plt.figure(figsize=(3, 3))

        final_pred = 0
        for _, frame_img in enumerate(video):
            frame_array = mx.nd.array(frame_img).expand_dims(axis=0).transpose((0, 3, 1, 2))
            pred = net(frame_array)
            final_pred += pred
        final_pred /= len(video)

        classes = net.classes
        top_classes = 5
        ind = nd.topk(final_pred, k=top_classes)
        print('\n \n ----------- \n \n')
        print('The input video is classified to be')
        for i in range(top_classes):
            top_ = classes[ind[i].asscalar()]
            prob_ = nd.softmax(final_pred)[ind[i]].asscalar()
            print(f'\t{top_}, with probability {prob_}.')

        # Select a frame to plot
        frame = video[0, :, :, :]
        frame_resize = cv2.resize(frame, (60, 60)) #pylint: disable=no-member
        plt.imshow(frame_resize)
        plt.title(classes[ind[0]], fontstyle='italic', backgroundcolor='pink', pad=10)
        counter += 1
        plt.axis('off')
        plt.show()
        counter += 1
        figure_counter += 1
    plt.savefig('./results/predictions.jpg', bbox_inches='tight')
    plt.show()
    print('\n \n')
    print('Done')

net_ = get_model('vgg16_ucf101', nclass=101, pretrained=True)
opt_dict = {'dir': '../data/ucf_sports_actions/frames/', 'img_format': '*.jpg',
            'resize': True, 'img_width': 224, 'img_height': 224, 'gray_scale': False, 
            'output': 'all_images.npy'}
opt_ = argparse.Namespace(**opt_dict)
frames_, _ , = get_images(opt_)
plot_predictions(frames_, net_)
# %%
