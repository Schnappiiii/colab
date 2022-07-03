import os, sys
import pandas as pd
import json
import numpy as np
# from icecream import ic
import shutil as sh

# Ignore warnings
import warnings
warnings.filterwarnings('ignore') 


def bbox(file_path):
    
    # open json file
    with open(file_path) as json_data:
        data = json.load(json_data)

    # extract the info in 'images'
    id_list = [i['id'] for i in data['images']]
    file_name_list = [i['file_name'] for i in data['images']]
    width_list = [i['width'] for i in data['images']]
    height_list = [i['height'] for i in data['images']]
    data_image = pd.DataFrame({'id':id_list, 'file_name':file_name_list, 'width':width_list, 'height':height_list})

    # extract the info in 'annotations'
    segmentation_list = [i['segmentation'] for i in data['annotations']]
    bbox_list = [i['bbox'] for i in data['annotations']]
    iscrowd_list = [i['iscrowd'] for i in data['annotations']]
    id_list = [i['id'] for i in data['annotations']]
    image_id_list = [i['image_id'] for i in data['annotations']]
    category_id_list = [i['category_id'] for i in data['annotations']]
    area_list = [i['area'] for i in data['annotations']]

    data_annotations = pd.DataFrame({'segmentation':segmentation_list,
                                    'bbox':bbox_list,
                                    'iscrowd':iscrowd_list, 
                                    'id':id_list, 
                                    'image_id':image_id_list, 
                                    'category_id':category_id_list, 
                                    'area':area_list})
    data_annotations['segmentation'] = data_annotations.segmentation.apply(lambda x: x[0])

    # merge these two dataframe
    file_name = []
    width = []
    height = []
    for i in data_annotations.image_id:
        fn = data_image[data_image.id.isin([i])].file_name.values  # <class 'numpy.ndarray'>
        file_name.append(''.join(fn))  # omit [''] 
        w = data_image[data_image.id.isin([i])].width.values 
        width.append(int(w))
        h = data_image[data_image.id.isin([i])].height.values
        height.append(int(h))

    data_annotations['file_name'] = file_name
    data_annotations['width'] = width
    data_annotations['height'] = height
    data_annotations = data_annotations[['file_name', 'image_id', 'id', 'width', 'height', 'bbox', 'segmentation', 'area', 'iscrowd', 'category_id']]

    # build data_box, preparing for label
    data_bbox= pd.DataFrame({})
    data_bbox['file_name'] = data_annotations.file_name
    data_bbox['image_id'] = data_annotations.image_id
    data_bbox['id'] = data_annotations.id
    data_bbox['x_min'] = data_annotations.segmentation.apply(lambda x: x[0]) 
    data_bbox['y_min'] = data_annotations.segmentation.apply(lambda x: x[1]) 
    data_bbox['x_max'] = data_annotations.segmentation.apply(lambda x: x[4]) 
    data_bbox['y_max'] = data_annotations.segmentation.apply(lambda x: x[5]) 
    data_bbox['width'] = data_annotations.bbox.apply(lambda x: x[2]) 
    data_bbox['height'] = data_annotations.bbox.apply(lambda x: x[3]) 

    return data_bbox



# Create bbox patch folder in order to train model.
# If you don't want to do this, no need to run the code below.
# If you would like to run this, just add these code before 'return data_box'
'''
    bbox_image_path = os.path.join(root_directory, 'bbox_image')

    # sh.rmtree(bbox_image_path)  # delete this folder
    # if file don't exist, then create
    try:
        os.makedirs(bbox_image_path)
    except:
        print ("Directory bbox_image_path already exist.")

    for i in data_bbox.id:
        tmp = data_annotations[data_annotations.id.isin([i])].file_name
        i_p = ' '.join(tmp)  # omit duplicated file name, '../...jpeg'
        img_path = os.path.join(root_directory, 'Data', 'Images', i_p)  # original data path
        file, ext = os.path.splitext(img_path)  # ('c:\\csv\\test', '.csv')

        im = Image.open(img_path)
        # Cropped image (It will not change orginal image) 
        one_row = data_bbox[data_bbox.id.isin([i])]

        im_ = im.crop((one_row.x_min, one_row.y_min, one_row.x_max, one_row.y_max)) 
        im_.save(os.path.join(bbox_image_path, str(i)+ext))  # , quality=95
    
'''