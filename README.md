# Data extraction from different types of charts in sustainabilty data reports
Most of charts in sustainabilty data reports cannot be detected using python PDF libraries. Since they are drawn with code, not simply copied images and pasted them on oages. Even if we can extract all the images in the pdf, but these images contain a variety of types, such as photos of people, landscape photos, signatures, logos, etc., later want to just extract the charts is very difficult.

Therefore, in this experiment we try to deploy YOLOv5 End-to-End object detector to detect different types of charts in images. After that for the different categories, data is extracted from the chart into a table.

## Extracting different types of charts
### Traning dataset
50 PDFs contain 5981 page images (selected from sustainabilty data report 2022-04-29). Only 715 out of 5981 pages contain charts, 5981/715=8.36 pages/chart. There are 1229 charts in 715 pages. Total images: 28714 (2000 generated images/class + 714 manual labeled page images) 

14 classes: 'single-legend vertical bar chart', 'multi-legend vertical bar chart', 'bi-directional vertical bar chart', 'single-legend horizontal bar chart', 'multi-legend horizontal bar chart', 'bi-directional horizontal bar chart', 'stacked vertical bar chart', 'stacked horizontal bar chart', 'line chart', 'muti-legend line chart', 'single pie chart', 'multi pie chart', 'single ring chart', 'multi ring chart'

<b> Image generation </b>
Firstly, using image augmentation, e.g., adding noise, resizing, shifting, rotating, etc. Then, selceting several pages without charts from data report as background (936 text pages in this experiment). Afterwards, randomly sample some points in this page, that is coordinate. Pasting this processed chart onto the page according to this coordinate. Hence one image is obtained with bounding box and don’t need to label them. We can generate as many images as we want by sampling points and using image augmentation. However, the drawback is obvious. Because the variation of pictures is not large, and the model is easy to over-fit.

### Training phase (TBD)
batch: 32, epoch: 30, 

### Test dataset (TBD)
99 PDFs contain ? images (selected from sustainabilty data report 2022-04-28)

### Test phase (TBD)
conf: 0.8, iou: 0.4. 221 charts were recognized. 

## Detecting the area of bar or pie or ring (TBD)
Applying YOLO Object Detection to detect the area of bar or pie or ring, in oder to seprate bar and legend, since they are the same color.

### Traning dataset (TBD)
FigureQA Dataset: https://www.microsoft.com/en-us/research/project/figureqa-dataset/
This dataset is very simple and the variation is small, so don't need to train lots of images. Just feeding 1000 charts per class into the model and around 10 epochs the model has already converged.

### Training phase (TBD)
batch: 16, epoch: 20, 

## Data extraction 
### Axis detection 



In this experiment, Google Coloud Vision OCR API was applied. Because it is more accurate than pytesseract in sustainabilty data report. Before runing Pipeline.ipynb, firstly deploy Google Coloud Vision OCR API. If using pytesseract, windows system user should download pytesseract.exe and set it in OCR.py. On the contrary, macOs system user don't need to do anything. (When using pytesseract library, different versions of pandas will generate different errors, this code has been verified on 1.3.5 and 1.5.3 versions)


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


