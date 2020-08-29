# Python Tensorflow Tutorial
Learning how to use TensorFlow with Python.

# Install python3 (if it doesn't exist)
```sudo apt install python3```

# Install pip3
```sudo apt install python3-dev python3-pip```

# Install virtualenv
```sudo pip3 install -U virtualenv```

# Setup virtualenv
To setup a virtualenv, use ```virtualenv --system-site-packages -p python3 ./venv```

To activate virtualenv, use ```source ./venv/bin/activate```

To deactive virtualenv, use ```deactivate```

To update packages in the virtualenv, use ```pip install --upgrade pip```

To list installed packages, use ```pip list```

# Install TensorFlow
In a virtualenv, do ```pip install --upgrade tensorflow```

# Install TensorFlow Hub and TensorFlow Datasets
To install TensorFlow Hub, use ```pip install  tensorflow-hub```

To install TensorFlow Datasets, use ```pip install -tfds-nightly```

# Install seaborn and tensorflow_docs
To install seaborn, use ```pip install seaborn```

To install tensorflow_docs, use ```pip install git+https://github.com/tensorflow/docs```

# Install pyyaml and h5py
To install pyyaml and h5py, use ```pip install pyyaml h5py```

# Install keras-tuner
To install keras-tuner, user ```pip install keras-tuner```