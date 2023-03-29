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
### Optical character recognition (OCR)
In this experiment, Google Coloud Vision OCR API was applied. Because it is more accurate than pytesseract in sustainabilty data report. Before runing Pipeline.ipynb, firstly deploy Google Coloud Vision OCR API and set "vision_api.json" file into OCR.py. If using pytesseract, windows system user should download pytesseract.exe and set it in OCR.py. On the contrary, macOs system user just need to set Google Vision API. (When using pytesseract library, different versions of pandas will generate different errors, this code has been verified on 1.3.5 and 1.5.3 versions)

Extracting useful data, i.e., text, xmin, ymin, xmax, ymax, width, height into dataframe format.

### Axis detection (TBD)
1. Firstly, the image is converted into black and white image (grayscale). (top left pixel coordinates!)
2. Scondly, searching for y-axis. Set search range (10, width//4). Sometimes there is a line on the leftmost of chart. Calculating the number of pixels less than 200 in each column. The maximum value is the x-coordinate of the y-axis.
3. Thirdly, searching for x-axis. If chart type is 'muti-legend', set search range (height-height//2, height-height//10), legends are under x-axis in most case in data report, else set search range (height-height//2, height-20), sometimes there is a line on the bottom of the picture. Calculating the number of pixels less than 200 in each row. The row maximum value is the y-coordinate of the x-axis. 
4. Fourthly, since some of charts have grid line, this can seriously affect the detection results. OCR result will be used for searching for axis. The range is the same above. Which column has the maximum value of ymin is y-axis and which row has the maximum value of xmax is x-axis. If no value is found, set it 0.
5. Fifthly, for y-axis, get the minimum of these two values. For x-axis, get the maximum of these two values. 

(result pictures)

## Title detection (TBD)
1. For title detection, don't need to sort dataframe, because Google Vision OCR is arranged in the order of the title.
2. If the distance of two words is not more than 20 or the distance of two line is not more than 10,, thus it's a part of title.  


## Axis ticks detection (TBD)
### X-ticks
1. Sort xmin, ymin by ascending order (Google Coloud Vision OCR API default order). 
2. Filter the text boxes which are below the x-axis (set a little tolerance 30) and to the right of y-axis. 
3. As one tick not always contain just one word, so if the distance of two words is not more than 20, considering it as a part of tick.

### Y-ticks
1. Sort xmin, ymin by ascending order (Google Coloud Vision OCR API default order). 
2. Filter the text boxes which are below title and above x-axis, as well as to the left of y-axis. (set a little tolerance 30) 
3. If there is a "%", value divided by 100. If ",", delete it, etc.

### Color isolation
A number of people suggest converting image to HSV or LAB color space, but they have already known a color. Some recommend using K-mean algorithm, prerequisite is we know the number of class. In our case the result of these two method is bad. 

I tried some image angmentation methods and think Gaussian Blur(radius=2)) is the best one. Making the blur slightly larger can effectively remove the text that has same color as bar or pie and also decrease the other small color noise, in order to increse the accuracy. Of course, not all the color in chart can be extracted, e.g., light color (light blue). The other preprocessing it's ok, like: adjust the color balance (0.9), diminish contrast (0.85), adjust image sharpness (1.5).

### Legend detection
1. Get contour of each color using blurred image. Then, deleting the bar, the rest of contour points belong to color.
2. Based on color legend contour, take backward values. Since one legend usually don't contain just one word, so if the distance of two words is not more than 20, considering it as a part of legend.

### Value detection
- value above the bar
According to bar ymin, set a interval to see if the value above the bar and within this interval. If each bar has its correspondng value, then don't need to detect y-axis.

- value don't above the bar
Based on y tick values using proportional relation to calculate the real value.
(at times y tick begin not from 0)

\begin{equation}
\begin{aligned}
y_value = \frac{y_pixel_max - pixel_list}{y_pixel_max - y_pixel_min} * interval + y_tick_min
\end{aligned}
\end{equation}




- pie or ring
After the color isolation, we can count the number of pixel that in this color area. Finally, divided by the total number and get the percentage. 


## Problem
- Chart detection (incomplete chart, such as missing legend or title, etc.) 

- Bar detection (training dataset is simple, if bar is very thin or there is no gap between the two bars, these bars cannot be or inaccurate recognized)

- Color extraction and isolation (If there are few components of a certain color, then it is difficult to extract and separate)
=======


=======


=======

Note:
* 


