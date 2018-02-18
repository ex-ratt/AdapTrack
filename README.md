AdapTrack
=========

This project provides SVM training, detection, and (adaptive) tracking via libraries and small demo applications.


Building the project
--------------------

### Prerequisites

* C++14
* [CMake](http://www.cmake.org/) >= 3.1
* [Boost](http://www.boost.org/) >= 1.48.0
* [OpenCV](http://opencv.org/) >= 2.4.3

### For usage

1. Create new project directory: `$ mkdir AdapTrack`
2. Enter the new directory: `$ cd AdapTrack`
3. Clone the repository (e.g. `$ git clone https://github.com/ex-ratt/AdapTrack.git` or using zip download), results in directory named `AdapTrack`
4. Create build directory next to `AdapTrack`: `$ mkdir build`
5. Change to build directory: `$ cd build`
6. Build project: `$ cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=../install ../AdapTrack/`
7. Compile the project: `$ make`
8. Install the libraries and binaries: `$ make install`

### For development with Eclipse

1. Create new project directory: `$ mkdir AdapTrack`
2. Enter the new directory: `$ cd AdapTrack`
3. Clone the repository (e.g. `$ git clone https://github.com/ex-ratt/AdapTrack.git` or using zip download), results in directory named `AdapTrack`
4. Create build directory next to `AdapTrack`: `$ mkdir build`
5. Change to build directory: `$ cd build`
6. Build Eclipse project: `$ cmake -G"Eclipse CDT4 - Unix Makefiles" -D CMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT=TRUE -D CMAKE_BUILD_TYPE=Release ../AdapTrack/`


Libraries
---------

* **libImageIO:** Loading and storing of images and annotations
* **libImageProcessing:** Image filtering, pyramid construction, feature extraction
* **libClassification:** Interfaces for binary and probabilistic classifiers and trainers, support vector machine, kernels
* **libSvm:** Wrapper around [libSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) that implements the classifier trainer interfaces of libClassification
* **libDetection:** Sliding-window-based detector and trainer
* **libTracking:** Adaptive model-free short-term single-target tracker and adaptive particle-filter-based multi-target tracker


Applications
------------

Some of the applications need annotations and configuration files. You can find examples of these files in the `resources` directory and in the description below.

### DetectorTrainer

Trains and tests detectors that are based on the sliding window technique, aggregated features, and a linear support vector machines.

#### Training a detector

`./DetectorTrainer train DIRECTORY IMAGES SETCOUNT [FEATURECONFIG TRAININGCONFIG]`

* DIRECTORY: directory that should be created (or re-used) and stores the config files and trained SVMs
* IMAGES: path to an XML file with image names and annotations, created with [dlib's](http://dlib.net/) imglab tool
* SETCOUNT: number of subsets to use for cross-validation (if set to 1, all images are used to create a single detector, so there won't be cross-validation)
* FEATURECONFIG: configuration file containing the feature parameters, see below for an example (only necessary if detector directory does not exist)
* TRAININGCONFIG: configuration file containing the training parameters, see below for an example (only necessary if detector directory does not exist)

Example: `$ ./DetectorTrainer train detector-fhog9-4x10 annotations.xml 4 features-fhog9-4x10 training-c10`

#### Testing a detector

`./DetectorTrainer test DIRECTORY IMAGES SETCOUNT DETECTIONCONFIG`

* DIRECTORY: directory that contains config files and trained SVMs (created when training the detector)
* IMAGES: path to an XML file with image names and annotations, created with [dlib's](http://dlib.net/) imglab tool
* SETCOUNT: number of subsets to use for cross-validation (if set to 1, all images are used to test a single detector, so there won't be cross-validation)
* DETECTIONCONFIG: configuration file containing the detection parameters, see below for an example

Example: `$ ./DetectorTrainer test detector-fhog9-4x10 annotations.xml 4 detection-40x40-5`

#### Showing detections

`./DetectorTrainer show DIRECTORY IMAGES SETCOUNT DETECTIONCONFIG [THRESHOLD]`

* DIRECTORY: directory that contains config files and trained SVMs (created when training the detector)
* IMAGES: path to an XML file with image names and annotations, created with [dlib's](http://dlib.net/) imglab tool
* SETCOUNT: number of subsets to use for cross-validation (if set to 1, all images are used to test a single detector, so there won't be cross-validation)
* DETECTIONCONFIG: configuration file containing the detection parameters, see below for an example
* THRESHOLD: SVM threshold (optional, defaults to 0.0)

Example: `$ ./DetectorTrainer show detector-fhog9-4x10 annotations.xml 4 detection-40x40-5 0.5`

#### Configuration files

Feature configuration

```
type fhog9                    ; feature type
                              ;   fhog# (Felzenszwalb's HOG variation, # = number of unsigned orientation bins)
                              ;   fpdw (features of the Fasted Pedestrian Detector in the West)
windowWidthInCells 10         ; width of the detection window in cells
windowHeightInCells 10        ; height of the detection window in cells
cellSizeInPixels 4            ; width and height of a cell in pixels
octaveLayerCount 10           ; number of image pyramid layers per octave - only used for training
```

Training configuration

```
mirrorTrainingData true       ; flag that indicates whether the training data is symmetric and should be mirrored to double the amount of data
maxNegatives 30000            ; maximum number of negatives to use - 0 for unlimited
randomNegativesPerImage 20    ; initial number of negative training examples per image
maxHardNegativesPerImage 100  ; maximum number of additional hard negatives sampled per image in each bootstrapping round
bootstrappingRounds 3         ; number of bootstrapping rounds
negativeScoreThreshold -1.0   ; hard negative training examples have an SVM score of at least this value
overlapThreshold 0.3          ; maximum allowed overlap between negative training examples and annotations
C 10                          ; SVM penalty multiplier
compensateImbalance true      ; flag that indicates whether to compensate for data imbalance
probabilistic false           ; flag that indicates whether to train a probabilistic SVM (computes and stores logistic parameters, does not influence detection)
```

Detection configuration

```
minWindowWidthInPixels 40     ; minimum width of objects to be detected
minWindowHeightInPixels 40    ; minimum height of objects to be detected
octaveLayerCount 5            ; number of image pyramid layers per octave
approximatePyramid false      ; flag that indicates whether to approximate all but one layer per octave
nmsOverlapThreshold 0.3       ; maximum allowed overlap between two different detections after non-maximum suppression
```

### SingleTracker

Tracks a single target without prior knowledge after initialization by the ground truth.

`./SingleTracker ANNOTATIONS BINS CELLSIZE TARGETSIZE PADDING ADAPTATION`

* ANNOTATIONS: path to an XML file with image names and annotations, created with [dlib's](http://dlib.net/) imglab tool
* BINS: number of unsigned orientation bins of the FHOG features
* CELLSIZE: width and height of the FHOG cells in pixels
* TARGETSIZE: size of the target in FHOG cells (larger one of width or height)
* PADDING: number of cells around the previous target position that is searched for the new position
* ADAPTATION: weight of the new SVM parameters between zero (no adpatation) and one (no memory)

Example: `$ ./SingleTracker annotations.xml 9 4 10 7 0.1`

### MultiTracker

Tracks multipe targets using a particle filter for each.

`./MultiTracker VIDEO SVM CELLSIZE DETECTIONTHRESHOLD VISIBILITYTHRESHOLD`

* VIDEO: camera device ID, video file, dlib annotation XML-file, or image directory
* SVM: text file that contains the SVM data (created by DetectorTrainer)
* CELLSIZE: width and height of the FHOG cells in pixels
* DETECTIONTHRESHOLD: SVM score threshold for detections to be reported
* VISIBILITYTHRESHOLD: SVM score threshold for tracks to be regarded visible

Example: `$ ./MultiTracker video.avi svm-fhog9-4x10 4 1.0 -0.25`


Resources
---------

There are some additional useful resource files in the directory `resources`:

* **svm:** Few SVMs trained by the DetectorTrainer on frontal heads
* **annotation:** Annotations of heads that can be used to train and test head detectors
* **features:** Exemplary feature parameter files
* **training:** Exemplary training parameter files
* **detection:** Exemplary detection parameter files

