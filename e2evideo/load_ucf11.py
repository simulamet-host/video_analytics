import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import our_utils
device = our_utils.get_device()

class VideoDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels

    def __getitem__(self, index):
        frame1 = self.frames[index]
        frame2 = self.frames[(index + 1) % len(self.frames)]
        label = self.labels[index]
        return frame1, frame2, label

    def __len__(self):
        return len(self.frames)

def get_data(labels_file, images_array):
    labels_list = []
    with open(labels_file, 'r') as f:
        for line in f:
            labels_list.append(line.strip())

    with np.load(images_array) as videos_frames:
        frames_array = videos_frames['arr_0']

    print('\n Splitting the data into training and test...\n')
    x_train, x_test, y_train, y_test=train_test_split(frames_array, labels_list, test_size=0.06,
                                                    random_state=10, shuffle=False)

    # pylint: disable=E1101
    x_train, x_test = torch.tensor(x_train).to(device), torch.tensor(x_test).to(device)
    train_dataset = VideoDataset(x_train, y_train)
    test_dataset = VideoDataset(x_test, y_test)
    return train_dataset, test_dataset