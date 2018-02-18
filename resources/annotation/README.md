Head Annotations
================

This directory contains annotations of heads within images. The corresponding images were taken from the [Pascal Visual Object Classes (VOC) Challenge](http://host.robots.ox.ac.uk/pascal/VOC/) and can be downloaded as part of the [VOC 2012 Development Kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) (see `JPEGImages` in the training/validation data archive). The annotations were created and can be edited with [dlib's](http://dlib.net/) imglab tool.

The annotations are rectangular axis-aligned bounding boxes that closely encapsulate the skull, ignoring hairstyle, hats, glasses, and other accessories. There are several annotation files, where each file contains the same annotated heads. They differ by the distinction between heads that are considered when training/testing and heads to ignore. The frontal head category, for example, only considers heads that are upright and frontal, while e.g. heads in profile view are ignored. Thus, they will be used as neither positive nor negative when training or testing a detector.

To make the annotated images work, the .xml files have to be placed next to a directory called `images` that contains the actual images. You can just extract `JPEGImages` from the Pascal VOC 2012 data set next to the annotations and rename the directory to `images`. Image files that are not mentioned in the annotations file are just ignored, as only a small subset of all the available images are annotated.

Categories/Files
----------------

The categories contain the same annotations of heads bigger than around 20 pixels, but consider and ignore different heads. The following list explains which heads are considered.

* **all.xml:** Real human heads that are in focus, completely within the image, and have more visible than occluded parts.
* **upright.xml:** All heads that are upright, meaning that they are not seen from below or above (roll/pitch angle of around 15° is allowed).
* **frontal.xml:** Upright heads that are seen from the front with a yaw angle below 45°.
* **profile.xml:** Upright heads that are seen from the side with a yaw angle between 45° and 90°.
* **frontal-profile.xml:** Upright heads that are seen from the front or the side with a yaw angle below 90°.
* **other.xml:** All heads that are neither frontal, nor profile.

