# context-aware-deepfake-detection

Midterm Report Website Link: https://github.gatech.edu/pages/dvsm3/context-aware-deepfake-detection/

/docs/ : Directory containing files for the Github Pages website with Jekyll theme

/pre_processing/pre_processing_demo.ipynb : ipython notebook containing a series of pre-processing steps done for the project.

## Repository Structure

### data_exp_notebooks
* data.ipynb
    * Organizes the data into two categories: Positive & Negative
    * Cleans data and eliminates duplicates
    * Handles missing data fields using unsupervised approaches
    * Creates directories /train, /val, /test to store videos
    * Creates the annotation csv files for training, validation and testing
        * Annotation video path files are created: contains video paths and corresponding labels (Used by dataloaders)
        * Annotation text files: contains cleaned text with corresponding labels
        * Annotation video files: contains video file names and corresponding labels

* analyze_data.ipynb
    * Performs Data Analysis on the datasets
    * Used to find corrupted files using OpenCV for further cleaning
    * Analyzes FPS and Duration for videos to determine frames to collect for training


* dataset.ipynb
    * Defines dataset class for Deepfake dataset
    * Validates loading using dataloader