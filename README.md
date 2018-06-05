
# Sklearn and Docker! And some Flask
A match made in heaven

__NOTE__: The docker image (as a .tar) won't be included as it is over 1GB in size


```python
__author__ = 'Alistair Rogers'
```

### Model Training and Persistence

With the stereotypical iris example...


```python
import flask
import pandas as pd
from sklearn import datasets
from sklearn import svm
```


```python
iris = datasets.load_iris()
```


```python
svm = svm.SVC()
```


```python
X, y = iris.data, iris.target
```


```python
X = pd.DataFrame(X)
X.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
```


```python
y = pd.DataFrame(y)
y.columns = ['Species']
```


```python
print(X.head())
```

       Sepal_Length  Sepal_Width  Petal_Length  Petal_Width
    0           5.1          3.5           1.4          0.2
    1           4.9          3.0           1.4          0.2
    2           4.7          3.2           1.3          0.2
    3           4.6          3.1           1.5          0.2
    4           5.0          3.6           1.4          0.2



```python
svm.fit(X, y)
```

    /Users/arogers2/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)





    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)



##### Serialise the model for scoring later on


```python
import pickle
pickle.dump(svm, open('svm_iris.pkl', 'wb'))
```

### Let's make a flask app for scoring on new data

Save the following as <b>app.py</b>

You can also create an empty __init.py__ if you'd like


```bash
%%bash
cat app.py
```

    from flask import Flask
    from flask import request
    from flask import jsonify
    from sklearn import svm
    import pickle
    import numpy as np
    import pandas as pd
    
    app = Flask(__name__)
    
    # Create a test method
    @app.route('/isalive')
    def index():
    	return "This API is Alive"
    
    @app.route('/prediction', methods=['POST', 'GET'])
    def get_prediction():
    
     # GET the JSONified Pandas dataframe
     print('Requesting...')
     json = request.args.get('data')
    
     # Transform JSON into Pandas DataFrame
     print('dataframing...')
     df = pd.read_json(json)
     df = df.reset_index(drop=True)
    
     # Read the serialised model
     print('reading model')
     modelname = 'svm_iris.pkl'
     print('Loading %s' % modelname)
     loaded_model = pickle.load(open(modelname, 'rb'), encoding='latin1')
    
     # Get predictions
     print('predicting')
     prediction = loaded_model.predict(df)
     prediction_df = pd.DataFrame(prediction)
     prediction_df.columns = ['Species']
     prediction_df.reset_index(drop=True)
    
     # OPTIONAL: Concatenate Predictions with original Dataframe
     df_with_preds = pd.concat([df, prediction_df], axis=1)
     return df_with_preds.to_json()
    
    if __name__ == '__main__':
     app.run(port=5000,host='0.0.0.0')
     #app.run(debug=True)


#### Check if the app works
Open a terminal window and navigate to the directory that you are working from. Then run __python docker/app.py__

You should get something like this (while debug mode is on..):
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)

We will also create a json to test the api endpoint. We will create this with the first 10 rows of the data that was used to train the model


```python
X.iloc[1:10, :].to_json()
```




    '{"Sepal_Length":{"1":4.9,"2":4.7,"3":4.6,"4":5.0,"5":5.4,"6":4.6,"7":5.0,"8":4.4,"9":4.9},"Sepal_Width":{"1":3.0,"2":3.2,"3":3.1,"4":3.6,"5":3.9,"6":3.4,"7":3.4,"8":2.9,"9":3.1},"Petal_Length":{"1":1.4,"2":1.3,"3":1.5,"4":1.4,"5":1.7,"6":1.4,"7":1.5,"8":1.4,"9":1.5},"Petal_Width":{"1":0.2,"2":0.2,"3":0.2,"4":0.2,"5":0.4,"6":0.3,"7":0.2,"8":0.2,"9":0.1}}'




```python
import requests
```

Let's try the testing isalive method first


```python
local_test = requests.get('http://0.0.0.0:5000/isalive')
```


```python
print(local_test.content)
```

    b'This API is Alive'


Now let's see what happens when we feed some data in.


```python
data = X.iloc[1:10, :].to_json()
```


```python
local_prediction = requests.get('http://0.0.0.0:5000/prediction?data='+data)
```


```python
local_prediction.text
```




    '{"Petal_Length":{"0":1.4,"1":1.3,"2":1.5,"3":1.4,"4":1.7,"5":1.4,"6":1.5,"7":1.4,"8":1.5},"Petal_Width":{"0":0.2,"1":0.2,"2":0.2,"3":0.2,"4":0.4,"5":0.3,"6":0.2,"7":0.2,"8":0.1},"Sepal_Length":{"0":4.9,"1":4.7,"2":4.6,"3":5.0,"4":5.4,"5":4.6,"6":5.0,"7":4.4,"8":4.9},"Sepal_Width":{"0":3.0,"1":3.2,"2":3.1,"3":3.6,"4":3.9,"5":3.4,"6":3.4,"7":2.9,"8":3.1},"Species":{"0":2,"1":2,"2":2,"3":2,"4":2,"5":2,"6":2,"7":2,"8":2}}'



In your terminal, you should see something like this:

Requesting... <br>
dataframing... <br>
reading model <br>
Loading svm_iris.pkl <br>
predicting <br>
127.0.0.1 - - [05/Jun/2018 09:28:32] "GET /prediction?data=%7B%22Sepal_Length%22:%7B%221%22:4.9,%222%22:4.7,%223%22:4.6,%224%22:<br>5.0,%225%22:5.4,%226%22:4.6,%227%22:5.0,%228%22:4.4,%229%22:4.9%7D,%22<br>Sepal_Width%22:%7B%221%22:3.0,%222%22:3.2,%223%22:3.1,%224%22:3.6,%225<br>%22:3.9,%226%22:3.4,%227%22:3.4,%228%22:2.9,%229%22:3.1%7D,%22Petal_Le<br>ngth%22:%7B%221%22:1.4,%222%22:1.3,%223%22:1.5,%224%22:1.4,%225%22:1.7<br>,%226%22:1.4,%227%22:1.5,%228%22:1.4,%229%22:1.5%7D,%22Petal_Width%22:<br>%7B%221%22:0.2,%222%22:0.2,%223%22:0.2,%224%22:0.2,%225%22:0.4,%226%22<br>:0.3,%227%22:0.2,%228%22:0.2,%229%22:0.1%7D%7D HTTP/1.1" 200


```python
print(pd.read_json(local_prediction.text))
```

       Petal_Length  Petal_Width  Sepal_Length  Sepal_Width  Species
    0           1.4          0.2           4.9          3.0        2
    1           1.3          0.2           4.7          3.2        2
    2           1.5          0.2           4.6          3.1        2
    3           1.4          0.2           5.0          3.6        2
    4           1.7          0.4           5.4          3.9        2
    5           1.4          0.3           4.6          3.4        2
    6           1.5          0.2           5.0          3.4        2
    7           1.4          0.2           4.4          2.9        2
    8           1.5          0.1           4.9          3.1        2


Well that all works, fantastic! Close your connection with a CTRL + C

## Building the Docker Image

There are two things you need to build a Docker image (besides Docker installed on your machine).

A Dockerfile
- Specifies the commands to run and sources dependencies.
<br>

A requirements file
- A list of all the dependencies you require, in this case it will be the python packages used

#### The Dockerfile


```bash
%%bash
cat Dockerfile
```

    FROM python:3.5.3
    MAINTAINER Alistair Rogers
    
    # Create a directory to work from
    WORKDIR /app/
    
    # Place the requirements file specifying all of the dependencies in that file
    COPY requirements.txt /app/
    RUN pip install -r ./requirements.txt
    
    # Place the Flask application and pickled model file into the directory
    COPY app.py __init__.py /app/
    COPY svm_iris.pkl /app/
    
    # Expose the app on port 5000
    EXPOSE 5000
    
    ENTRYPOINT python ./app.py


#### The Requirements File


```bash
%%bash
cat requirements.txt
```

    numpy==1.13
    scipy==0.19.1
    Flask==0.12.2
    scikit_learn==0.18.1
    pandas==0.18.1


##### Build the Docker Image!!

You can do this with the following command: <br>
<b>docker build . -t [NAME]</b>

We will call this image __iris_svm__


```python
!docker build . -t iris_svm
```

    Sending build context to Docker daemon  1.193GB
    Step 1/9 : FROM python:3.5.3
     ---> 56b15234ac1d
    Step 2/9 : MAINTAINER Alistair Rogers
     ---> Using cache
     ---> e8948cd02846
    Step 3/9 : WORKDIR /app/
     ---> Using cache
     ---> 2aa2e410bb64
    Step 4/9 : COPY requirements.txt /app/
     ---> Using cache
     ---> c1b090adbb9d
    Step 5/9 : RUN pip install -r ./requirements.txt
     ---> Using cache
     ---> cee84e2a220b
    Step 6/9 : COPY app.py __init__.py /app/
     ---> Using cache
     ---> 159569d5d750
    Step 7/9 : COPY svm_iris.pkl /app/
     ---> Using cache
     ---> 89dc1650ecee
    Step 8/9 : EXPOSE 5000
     ---> Using cache
     ---> 4475acdeb6ab
    Step 9/9 : ENTRYPOINT python ./app.py
     ---> Using cache
     ---> a644e6b2467e
    Successfully built a644e6b2467e
    Successfully tagged iris_svm:latest


Now we can save this docker image as a .tar file and do some other stuff with it. We could send it to someone else who has docker so they can run it or use a cloud service to expose the model as a service etc


```python
!docker save iris_svm > iris_svm.tar
```

We can see that the tar file is ready and available


```python
!ls 
```

    Dockerfile                __init__.py               requirements.txt
    README.md                 app.py                    svm_iris.pkl
    Sklearn_with_Docker.ipynb iris_svm.tar


## Running the Docker Image

In order to run a docker image from a tar file, we must first load it:
    <b>docker load -i NAME.tar</b>


```python
!docker load -i iris_svm.tar
```

    Loaded image: iris_svm:latest


If we try to run the app on port 5000 (as specified in the flask app) then message we get when we run it will say it's listening on port 5000, but if we try to access that on the same host we get an error.

So need to expose the port outside the container that it's running in

Let's run it on ports 5001:5000

Run this on your command line! 
<b>docker run -p 5001:5000 -it iris_svm</b>
We can't run it here because it would continuously run and no other cell could be executed in this notebook

You should get a similar output as before:

Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)





```python
request_docker_test = requests.get('http://localhost:5001/isalive')
request_docker_test.text
```




    'This API is Alive'



Lets try this with the same data as before


```python
request_docker_pred = requests.get('http://localhost:5001/prediction?data='+data)
request_docker_pred.text
```




    '{"Petal_Length":{"0":1.4,"1":1.3,"2":1.5,"3":1.4,"4":1.7,"5":1.4,"6":1.5,"7":1.4,"8":1.5},"Petal_Width":{"0":0.2,"1":0.2,"2":0.2,"3":0.2,"4":0.4,"5":0.3,"6":0.2,"7":0.2,"8":0.1},"Sepal_Length":{"0":4.9,"1":4.7,"2":4.6,"3":5.0,"4":5.4,"5":4.6,"6":5.0,"7":4.4,"8":4.9},"Sepal_Width":{"0":3.0,"1":3.2,"2":3.1,"3":3.6,"4":3.9,"5":3.4,"6":3.4,"7":2.9,"8":3.1},"Species":{"0":2,"1":2,"2":2,"3":2,"4":2,"5":2,"6":2,"7":2,"8":2}}'




```python
print(pd.read_json(request_docker_pred.text))
```

       Petal_Length  Petal_Width  Sepal_Length  Sepal_Width  Species
    0           1.4          0.2           4.9          3.0        2
    1           1.3          0.2           4.7          3.2        2
    2           1.5          0.2           4.6          3.1        2
    3           1.4          0.2           5.0          3.6        2
    4           1.7          0.4           5.4          3.9        2
    5           1.4          0.3           4.6          3.4        2
    6           1.5          0.2           5.0          3.4        2
    7           1.4          0.2           4.4          2.9        2
    8           1.5          0.1           4.9          3.1        2


Woohoo! 

Now CTRL + C to stop your docker
