from bbox_folder import bbox
import os, sys
import pandas as pd
import json
import numpy as np
# from icecream import ic
import shutil as sh

from matplotlib import pyplot as plt 
from matplotlib.patches import Rectangle
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings('ignore') 


# if your json file is not this path, just change root_directory
root_directory = sys.path[0]  # current directory / data folder that includes Data: 'Images'&'Annotations'
annotation_path = os.path.join(root_directory, 'Data', 'Annotations', 'annotated_functional_test3_fixed.json')

data_bbox = bbox(annotation_path)  # get data_box dataframe

label_dict = {'1': 'Rim', '2': 'Scratch', '3': 'Dent', '4': 'Other'}   # dictionary{label:name} 
save_label = pd.DataFrame(columns=['image_id', 'id', 'human_label', 'label_name'])  # build empty dataframe

tmp = data_bbox.iloc[:5,:]  # the number of images that you want to label
for i,name in tmp.iterrows():

    # get one row in data_bbox
    [file_name, image_id, id, left, top, right, bottom, width, height] = name
    img_path = os.path.join(root_directory, 'Data', 'Images', file_name)

    # prompt
    print(f'current image_id is {image_id} and current id is {id}')
    print('1: Rim, 2: Scratch, 3: Dent, 4: Other')

    # draw original image and bbox patch
    plt.ion()  # Turn the interactive mode on.
    figure, axis = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [6, 1]}, figsize=(15,6), dpi=120)
    img1 = Image.open(img_path)
    img2 = img1.crop((left, top, right, bottom))
    axis[0].imshow(img1)
    axis[0].add_patch(Rectangle((left,top), width, height, fc='none', color='red', linewidth=1, linestyle="-"))
    axis[0].set_title("original image")
    axis[1].imshow(img2)
    axis[1].set_title("bbox patch")

    # type label in dataframe
    label = int(input("Please input a number: "))  # enter a number
    save_label.loc[i] = [image_id, id, label, label_dict[str(label)]]

    plt.cla()  # clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
    plt.clf()  # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
    plt.close()  # closes a window, which will be the current window, if not specified otherwise.

# save dataframe in csv and json
save_label_path = os.path.join(root_directory, 'save_label.csv')
save_label_path1 = os.path.join(root_directory, 'save_label.json')
save_label.to_csv(save_label_path, index=False, header=True)
save_label.to_json(save_label_path1, orient='records')
print("file has already saved.")
