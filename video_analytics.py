"""
Domain_specific module. It contains functions that can be applied,
to only specific dataset. This initial version is only applicable to the
hand washing dataset.
"""
import pandas as pd
import glob
import image_preprocessing

images_ = image_preprocessing.get_images()
print(images_.shape)

data_summary = pd.read_csv('./data/Dataset1/statistics.csv')
print(data_summary)