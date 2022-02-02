# OC_image_segmentation
This project was carried out as part of the training of the OC IA engineer.

## Context

Future Vision Transport is a company that develops in-vehicle computer vision systems for autonomous vehicles.
The different parts of the system:
1.	Real-time image acquisition
2.	Image processing
3.	Image segmentation 
4.	Decision system
This project concerns the image segmentation part (3) which is fed by the image processing block (2) and which feeds the decision system (4).


## Goal
The objective of this project is to create a first image segmentation model that will be easily integrated into the complete chain of the embedded system.

<img width="682" alt="image" src="https://user-images.githubusercontent.com/66125524/152153876-811c2221-4844-45b5-97df-c4539a806565.png">
 

## Deliverables:

 - Scripts developed on Azure Machine Learning to run the full pipeline. 

 - A Flask API deployed via the Azure service that will receive as input the identifier of an image and return the image with the segments identified by your model and the image with the identified segments annotated in the dataset. 

 - A technical note containing a presentation of the different approaches. 

 - PowerPoint presentation

 - The dataset is provided by [CITYSCAPES](https://www.cityscapes-dataset.com/dataset-overview/)
