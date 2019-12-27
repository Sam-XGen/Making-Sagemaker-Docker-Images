import os
import shutil
import argparse

#creating argument parser
parser = argparse.ArgumentParser()

#add an argument to be used
parser.add_argument('--input_model_name', type=str)

#creating object to use arguments in the script
args, _ = parser.parse_known_args()


#set the working directory
#if os.getcwd() != '/home/ec2-user/environment/building-docker/':
    #os.chdir(os.getcwd() + '/building-docker/')

def copy_move_rename(source_dir, old_file_name, destination_dir, new_file_name=None):
    '''
    This function moves, copies, and renames a file
    '''
    if new_file_name == None:
        new_file_name = old_file_name
    
    #what file is going to be moved
    source_file = os.path.join(os.curdir, source_dir, old_file_name)
    #copy file to new location
    shutil.copy(source_file, destination_dir)
    #copied file in new location
    destination_file = os.path.join(os.curdir, destination_dir, old_file_name)
    #new file name
    new_destination_file_name = os.path.join(os.curdir, destination_dir, new_file_name)
    #rename file
    os.rename(destination_file, new_destination_file_name)
    
    print(old_file_name + ' was moved from ' + source_dir + ' to ' + 
        destination_dir + ' and renamed to ' + new_file_name)



#model options
model_folder_name = args.input_model_name.replace('-', '_')
model_folder_name = model_folder_name[10:]


#regression 
#model_folder_name = 'linear_regression'
#model_folder_name = 'ridge_regression'
#model_folder_name = 'lasso_regression'
#model_folder_name = 'logistic_regression'

# naive bayes
#model_folder_name = 'naive_bayes_multinomial'

# support vector machines
#model_folder_name = 'support_vector_classification'
#model_folder_name = 'support_vector_regression'


# stochastic gradient descent
#model_folder_name = 'stochastic_gradient_descent_classification'
#model_folder_name = 'stochastic_gradient_descent_regression'



# k-nearest neighbors
#model_folder_name = 'k_nearest_neighbors_classification'
#model_folder_name = 'k_nearest_neighbors_regression'


# decision tree
#model_folder_name = 'decision_tree_classification'
#model_folder_name = 'decision_tree_regression'


# random forest
#model_folder_name = 'random_forest_classification'
#model_folder_name = 'random_forest_regression'


# gradient boosting
#model_folder_name = 'gradient_boosting_classification'
#model_folder_name = 'gradient_boosting_regression'


# basic neural network
#model_folder_name = 'basic_neural_network_classification'
#model_folder_name = 'basic_neural_network_regression'








docker = """# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:16.04

MAINTAINER Amazon AI <sage-learner@amazon.com>


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && \
    pip install numpy==1.16.2 scipy==1.2.1 scikit-learn==0.20.2 pandas flask gevent gunicorn && \
        (cd /usr/local/lib/python2.7/dist-packages/scipy/.libs; rm *; ln ../../numpy/.libs/* .) && \
        rm -rf /root/.cache

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY """ + model_folder_name + """ /opt/program
WORKDIR /opt/program"""



#open Dockerfile file
file = open('Dockerfile', 'w')
#write query to file
file.write(docker)
#close and save file
file.close()


build_and_push = """#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

chmod +x """ + model_folder_name + """/train
chmod +x """ + model_folder_name + """/serve

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build  -t ${image} .
docker tag ${image} ${fullname}

docker push ${fullname}"""

#open text file
file = open('build_and_push.sh', 'w')
#write query to file
file.write(build_and_push)
#close and save file
file.close()


#make a new directory
try:
    os.mkdir(model_folder_name)
except:
    print('Directory already exists. Moving to next step.')


# directory path for saving files later
inside_model_folder = os.path.join(os.curdir, model_folder_name)

#give the model a name
model_name = model_folder_name.replace('_', '-')

#make predicting file
predictor = """# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback

import flask
import scipy
import pandas as pd

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        '''Get the model object for this instance, loading it if it's not already loaded.'''
        if cls.model == None:
            with open(os.path.join(model_path, '""" + model_name + """-model.pkl'), 'r') as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input):
        '''For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe'''
        clf = cls.get_model()
        return clf.predict(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    '''Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully.'''
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='  ', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    '''Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    '''
    data = None

    print('Convert to Pandas')
    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        print('Flask utf-8')
        data = flask.request.data.decode('utf-8')
        print('StringIO')
        s = StringIO.StringIO(data)
        
        data = pd.read_csv(s, header=None)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    predictions = ScoringService.predict(data)

    # Convert from numpy back to CSV
    out = StringIO.StringIO()
    pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')"""


#open text file
file = open(os.path.join(inside_model_folder, 'predictor.py'), 'w')
#write query to file
file.write(predictor)
#close and save file
file.close()



#copy move and rename train file
copy_move_rename('train_scripts', 'train-' + model_name, model_folder_name, 'train')

#copy move and rename nginx file
copy_move_rename('train_scripts', 'nginx.conf', model_folder_name)

#copy move and rename serve file
copy_move_rename('train_scripts', 'serve', model_folder_name)

#copy move and rename wsgi file
copy_move_rename('train_scripts', 'wsgi.py', model_folder_name)

