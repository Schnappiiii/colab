=======

The first step: try AutoEncoder to classify the pictures with unsupervised learning method.

Run AutoEncoder.ipynb. The rim images can be detected but other categories not, so the result is not well and we should deploy active learning method. 

=======

The second step: label all the images with bounding box manually

Using bbox_folder.py to create folders for different kinds of images with bounding box based on the given dataset. Then, run label.py to label the images. (4 classes: dent, rim, scratch, other)

(Run data_processing.ipynb to bring the information from the original data set together)

=======

The third step: try to use GAN to increse the number of images.

Run GAN_bbox.ipynb and the results don't seem to be looking very good. Therefore, this way does not work.

=======

The fourth step: using different models to train the data and compare them.

1. KNN accuracy: 0.44. Run KNN.ipynb.

2. CNN accuracy: 0.68. Run CNN.ipynb.
CNN with validation dataset: accuracy 0.66. Run CNN_validataion.ipynb.

3. ResNet50 accuracy: 0.84. Run ResNet50.ipynb.
ResNet50 with validation dataset: accuracy 0.777. Run ResNet50_validataion.ipynb.

4. DenseNet201 with validation dataset: accuracy 0.81. Run DenseNet201_validataion.ipynb.

5. EfficientNetB7 with validation dataset: accuracy 0.79. Run EfficientNetB7_validataion.ipynb. 

=======

The fifth step: write the report about the algorithm part.

=======

Note:
* the class folders of Annotated_images are not uploaded, since the size is not small.


