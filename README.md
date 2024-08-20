# IML DataSoft Python REST API Project

Deploy methods of IML DataSoft Python REST API project are given below.

## Introduction And Installation Guide

- Introduction: https://www.youtube.com/watch?v=RcLk0Nz2ZSk
- Installation Guide: https://www.youtube.com/watch?v=_S2LX0eOtWI

### Project Publishing Instructions
The project can be published in two different ways.
* Docker
* Pip

#### Installation with Docker
To publish the project on Docker, go to the project directory and write the following codes;

```

  # deploy project with docker
  $ docker-compose up

  # compile docker files
  $ docker-compose build
  
```

#### Installation with Pip
To publish the project in Pip, go to the project directory and write the following codes;

```

  # package install
  $ pip install -r requirements.txt

  # python start
  $ flask run

  # rabbitmq consumer start
  $ python3 consumer.py

  # project start
  $ bash start.sh
  
```

### Project Information
It is the Python REST API of the IML DataSoft Project, connected to the Hasura backend. There are Simulator and AI Hub in the Python API. It is supported with Flask structure and queue structure is established with rabbitmq.

#### Simulator
It is triggered when a record is added to the simulators table in the backend and generates synthetic data containing selected error types.

```

  # Generated Data Types
  - Sound
  - Vibration
  - Temperature

  
```

#### AI Hub
It is triggered when a record is added to the algorithm_results table. It runs the selected data set and selected deep learning algorithms. It queues all deep learning run requests in the incoming trigger with rabbitmq and runs them sequentially via consumer.

```

  # Deep Learning Algorithms
  - ABOD Outlier Detection
  - Autoencoder Outlier Detection
  - CBLOF Outlier Detection
  - Deep Autoencoder Outlier Detection
  - Ensemble Outlier Detection
  - Feature Bagging Outlier Detection
  - GMM Outlier Detection
  - GRU Anomaly Detection
  - HOBS Outlier Detection
  - Isolation Forest
  - K-Means Outlier Detection
  - KNN Outlier Detection
  - Local Outlier Factor
  - LSTM Anomaly Detection
  - LSTM Autoencoder
  - PCA Outlier Detection
  - Robust Covariance Outlier Detection
  - RNN Anomaly Detection
  - XGBoost Outlier Detection
  
```
