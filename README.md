___
# CricVision : Unleashing Cricket Insights with AI Swing
___

## Unlocking cricket insights through machine learning! Analyze Sri Lanka’s international matches, predict player impact, and gain valuable insights for teams and fans. 🏏🔍

## Overview

Welcome to CricVision, an innovative project aimed at unlocking cricket insights through machine learning. Our platform allows for the analysis of Sri Lanka’s international matches, prediction of player impact, and generation of valuable insights for teams and fans alike. 🏏🔍

## Features

- **Match Analysis**: Detailed breakdowns of matches, including performance metrics for batters, bowlers, and fielders.
- **Player Impact Predictions**: Predictive analytics to forecast the impact of individual players on the outcome of matches.
- **Insight Generation**: Advanced insights for strategic planning and fan engagement.
- **Interactive Visualizations**: User-friendly visual representations of data for easy interpretation.

## Project Structure

The project is organized as follows:

- `Data/`: Contains the datasets used for analysis and model training.
  - `raw_data/`: Raw data collected from various sources.
  - `processed_data/`: Data that has been cleaned and processed for analysis.
  - `selected_data/`: Specific datasets selected for detailed analysis.
- `Models/`: Contains the machine learning models used for predictions.
- `EDA/`: Exploratory Data Analysis notebooks and scripts.
  - `Win_predictor.ipynb`: Jupyter notebook for win prediction model development.
- `scripts/`: Contains Python scripts for data processing, model training, and evaluation.
- `tests/`: Unit tests to ensure the integrity and performance of the models and scripts.
- `Dockerfile`: Configuration file for Docker containerization.
- `requirements.txt`: List of dependencies required to run the project.
- `README.md`: Project documentation and overview.

## Installation

To run this project locally, follow these steps:

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/cepdnaclk/e19-co544-cricket-analytics-and-prediction.git
   cd e19-co544-cricket-analytics-and-prediction

2. **Set Up the Virtual Environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. **Install Dependencies:**
   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt




# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app

