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

	
## Content





