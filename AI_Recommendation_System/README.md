# AI Tech Stack Recommender

## Project Overview
The AI Tech Stack Recommender is a Python-based recommendation system that suggests suitable career roles based on a user's technical skills.

## Features
- Accepts user skills as input.
- Uses TF-IDF vectorization to represent skills numerically.
- Calculates cosine similarity between user skills and career skill requirements.
- Ranks career roles by similarity score.
- Displays the top 3 career recommendations.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity

## Dataset
The `raw_skills.csv` file contains career roles and their associated technical skills.

## How to Run
1. Install the required libraries:
   `pip install pandas scikit-learn`
2. Run the application:
   `python main.py`
3. Enter at least three different skills separated by commas.

## Project Status
The basic recommendation system has been implemented and tested with the sample dataset.