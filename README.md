# Leaf segmentation project 2019

This project was to create a pipeline for the segmentation of leaves and leaf lesions using the fastai api. The aim is to allow quantitative measurement
of lesion sizes and degree of chlorosis (leaf greenness), and hence quantitative analysis of plant disease progression. The pipeline first takes in
an image and feeds it into the leaf segmentation model (Black pixels denote background, and white pixels are predicted leaf areas).

<img src="https://github.com/kkl116/Leaf_segmentation/blob/main/assets/leaf_seg_example.png" width="710" height="563">

Leaves in the image are segmented, and each object is given an id. K-means clustering is then used to group leaves together according to settings provided
in notebook (e.g. group by row/columns, and how many expected classes there are). Each segmentation is used to extract individual leaf images. Each image
is then fed into the lesion segmentation model, creating a prediction of the lesion areas (Black denotes backgrond, grey denotes leaf areas, and white denotes predicted lesion areas). 

<img src="https://github.com/kkl116/Leaf_segmentation/blob/main/assets/lesion_seg_example.png" width="302" height="284">


