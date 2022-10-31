# Docker for Machine Learning

#### Here we have a simple AI solution using Docker. The application is built from scratch and aims to generate predictions and receive numerical data as input.

## **Tree**

```bash
C:.
│   README.md
│
├───App
│   │   app_inference.py
│   │   app_training.py
│   │   docker-compose-inference.yml
│   │   docker-compose.yml
│   │   Dockerfile
│   │   Dockerfile_inference
│   │   requirements.txt
│   │
│   ├───Inference
│   │   │   inference.py
│   │   │
│   │   └───__pycache__
│   │           inference.cpython-39.pyc
│   │
│   ├───Model
│   │       model.joblib
│   │
│   ├───Results
│   │       predicted_values.npy
│   │
│   ├───static
│   │   └───images
│   │           diabetes.jpg
│   │           docker.jpg
│   │
│   ├───templates
│   │       generated.html
│   │       generated_inference.html
│   │       home.html
│   │       home_inference.html
│   │       inference.html
│   │       training.html
│   │
│   └───Training
│       │   generating_data.py
│       │   train.py
│       │   train_online.py
│       │
│       └───__pycache__
│               train_online.cpython-37.pyc
│               train_online.cpython-39.pyc
│
└───Imgs
        docker_inference.png
        docker_training.png
        overview.png
        readme.MD
        training_finished.png
        training_home.png
        training_url.png
```

## **Data**

#### We are using the diabetes dataset from sklearn. We have a total of 442 instances. The data has ten features:

#### &emsp; **Age** (in years)
#### &emsp; **Sex**
#### &emsp; **Bmi** (body mass index)
#### &emsp; **Bp** (average blood pressure)
#### &emsp; **S1** (tc, total serum cholesterol)
#### &emsp; **S2** (LDL, low-density lipoproteins)
#### &emsp; **S3** (HDL, high-density lipoproteins)
#### &emsp; **S4** (tch, total cholesterol / HDL)
#### &emsp; **S5** (ltg, possibly log of serum triglycerides level)
#### &emsp; **S6** (glu, blood sugar level)

#### The output is a numerical value related to the "quantitative measure of disease progression one year after baseline." More information in the [Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset).


## **Model**
#### We use linear regression to perform the predictions. We divide the dataset into two parts: training and testing. We are not using validation data for hyperparameter tuning. Our goal is to show how to build an AI software product using Docker. Furthermore, linear regression is not interesting when we consider hyperparameter tuning.

## **Inference**
#### We will load the trained model and make inferences. All the data is available online as .csv files. 

## **Overview**

<p align="center">
  <img src="Imgs\overview.png" />
</p>

#### The data available online is loaded in a web app, this app has an interface in which the user is asked to give a link with the data. With the data available, the training is performed, and a model is generated. In our case, we create a volume using Docker compose to make this model available in the models' folder. This folder is shared with the Inference container.

#### We have two containers, Training and Inference. The Training container is a web application used to generate our Machine Learning model. The Inference container loads the Machine Learning model generated by Training container and perform the prediction after receives the data. The prediction is a numpy array saved in .npy format and available in another volume called Results.

## **Training container**

### More details:

<p align="center">
  <img src="Imgs\docker_training.png" />
</p>

## **Inference container**

### More details:

<p align="center">
  <img src="Imgs\docker_inference.png" />
</p>

## **Docker**

#### When we use docker, we need to build an **image** and using this image, we are able to run a **container** with our application. We use **Dockerfile** to create the Docker image and **docker-compose.yml** to create and run the container.

## **Web application**
### Home
<p align="center">
  <img src="Imgs\training_home.png" />
</p>

### Put the URL
<p align="center">
  <img src="Imgs\training_url.png" />
</p>

### Message of finished inference
<p align="center">
  <img src="Imgs\training_finished.png" />
</p>

## **Usage**

### For creating the Training image:
```bash
docker build -f Dockerfile .
```  
### For creating the Inference image:
```bash
docker build -f Dockerfile_inference .
```  

### For running the Training docker container:
```bash
docker-compose -f docker-compose.yml up
```  
### For running the Inference docker container:
```bash
docker-compose -f docker-compose-inference.yml up
```  
