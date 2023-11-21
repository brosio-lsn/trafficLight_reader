# Traffic Light Reader

## Overview
The Traffic Light Reader is an advanced machine learning-based application designed to detect and interpret traffic light signals from images. It employs image processing techniques to identify traffic light boxes, applies transformations to simplify the images, and then detects circles within each masked area to determine the dominant color among red, orange, and green.

## Key Features
- Traffic light box detection using machine learning.
- Image transformation for simplification.
- Circle detection in masked areas.
- Color analysis to identify the dominant color among red, orange, and green.

## Technical Details
- The application first masks each detected traffic light box.
- Applies image transformations to simplify the traffic light image for easier processing.
- Detects circles within each masked box.
- Analyzes each circle to determine if red, orange, or green is the dominant color.

## Result Image
![Traffic Light Detection Result](https://github.com/brosio-lsn/trafficLight_reader/blob/master/result.jpg)

## Usage
Instructions on how to use the application, including image input requirements and execution steps, are provided in the project's documentation.

## Acknowledgments
- Special thanks to contributors and machine learning researchers who inspired and supported the development of this project.
