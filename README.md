## Skyebase_concrete_cracks

## Table of contents
* [General info](#general-info)
* [Results](#results)
* [Technologies](#technologies)
* [Content](#content)

## General info
The goal of the project is to detect cracks in concrete above and under water. 
And the detection of crack width and length and surface area.

## Results

- make opencv draw a contour aroun the cracks, getting the surface area of the crack with the contourArea function.
- segment the crack from the background based on a RGB treshold, being able to get the surface area based on RGB pixel values.
- draw a bounding box around the crack area based on the contour x&y coordinates, but could not automatise it with cv2 for all images.
- use opencv to life detect cracks based on threshold pixel values. 
- detect crack objects using yolo, opencv and Darknet, drawing a boundingbox around the crack object
- trained a CNN model based on open source data from Kaggle, accuracy was only 50% (should probably used pretrained CNN)

## Technologies
- CNN
- Darknet
- opencv
- yolov3
- labelimg

	
## Content
- cracks.py -> loading in the data
- train_CNN_model.py -> training the data with CNN model
- contour.py -> drawing contour around cracks in images
- contour_area.py -> saving surface areas of cracks in df
- crack_pixel_counter.py -> saving surface areas of cracks in df
- irl_object_contours.py -> detecting cracks irl with camera
- object_detection_test.py -> training model to detect crack, draw boundingboxes around it
- google_collab_train_darknet.py -> train darkent pretrained CNN with crack images, where i manually labeled the images with 'labelimg'




