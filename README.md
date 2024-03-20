# VK intern Retail Geoanalytics

## Project Structure
The project includes the following key components:



`data/train.csv`: The training dataset containing unique identifiers for retail outlets, their coordinates (latitude and longitude), and the target variable (success of the outlet)

`data/test.csv`: The test dataset containing unique identifiers for retail outlets and their coordinates for success prediction

`data/features.csv`: Additional features for the retail outlets that can be used to improve prediction quality

`data/sample_submission.csv`: An Example of submission file

`submission.csv`: Submission file

`requirements.txt`: A file listing the dependencies required for the project

`solution.py`: A script for generating predictions

`train.py`: A script for training the model

`features.py` A script for making features

## Getting Started
To get started with the project, follow these steps:

Clone the project repository to your local machine.
Install the necessary dependencies using the requirements.txt file:

`pip install -r requirements.txt`

Run the solution.py script to train the model and generate predictions:

`python solution.py`

## Results
After running the solution.py script, a file with predictions will be saved in the project's root directory named `submission.csv`. In this file, a predicted success score will be provided for each outlet in the test dataset.
