# RAMQA PROJECT
## Table of content
-[Overview](#overview)
-[Dataset Description](#dataset-description)
-[Project Workflow](#project-workflow)
-[Conclusion](#conclusion)

## Overview
Our project, ** RAMQA**  relies on eye-tracking technology to enable users to control a computer mouse using only their eye movements, without the need for hand movements. The primary objective of our project is to assist individuals with paralysis or severe mobility impairments in using technology, interacting with computers, and living their lives more independently. The application of this technology significantly improves the quality of life for these individuals, providing them with the ability to communicate and perform daily tasks with ease.Overall, through RAMQA we aspire to offer an advanced technological solution that opens new possibilities for people with special needs and enhances the role of technology in improving human life

## Dataset Description

Initially, we experimented with various pre-existing datasets, including GazeCapture, Synthetic Gaze and Face Segmentation datasets, and EVE (End-to-end Video-based Eye- tracking) datasets. However, we encountered challenges due to the enormous size of these datasets, such as GazeCapture, which includes data from 1,450 individuals and approximately 2.5 million frames. Due to these constraints, we adapted our collection methods and decided to create our own datasets. In our approach, we create our datasets by create a code that saved the results of coordinates with each selected points of landmark points in CSV files, then used PyCaret to evaluate different machine learning models, aiming to learn from the relationships between the (X, Y) coordinates of the mouse and facial landmarks. This method allowed us to manage data size more effectively and tailor our dataset to the specific needs of our project.


## Conclusion
RAMQA project enhances the lives of individuals with severe mobility impairments by leveraging advanced eye-tracking technology. It empowers users to control digital interfaces through eye movements, increasing their independence and social integration while improving their overall quality of life.
