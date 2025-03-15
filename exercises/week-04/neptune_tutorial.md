```py
# -*- coding: utf-8 -*-
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
```

# Neptune.AI logging tutorial
#### Author: Bc. Marek Adamove (2025)
----

## About Neptune.AI

[Neptune.AI](https://neptune.ai) is an experiment tracker, that allows researchers and their teams to track their experiments, trainings, results and artifacts created during the experimentation.

## Setting up
### Installation

To install the Neptune module in your Python environment use the following pip command:

```bash
pip install neptune
```

### Project creation

On the website, create a new project for your experiment. One project can hold multiple experiments.

### Project import

To use Neptune in your code, first import the module and initialize a run as follows:

```python
import neptune

# Run initialization
run = neptune.init_run(
    project="DL_projects/Experiment01", # Your project name
    api_token=os.getenv("NEPTUNE_API_TOKEN"), # API login token
    tags=["LSTM", "Attention"], # You can put any tags here (optional)
    description="Trying new LSTM and attention architecture" # Some description (optional)
)
```

## Logging
### Simple variable

If you want to log single variable you can do so like that:

```python
run["learning_rate"] = 0.01
```

### Dictionary

You can also log your experiments settings using dictionary:

```python
params = {
    "learning_rate": 0.01,
    "optimizer": "Adam",
    "batch_size": 128,
    "use_batchnorm": True
}

# Take run key as a pathing system
# For example settings/config == directory settings -> file config
run["settings/config"] = params 
```

### Arrays and lists

You can also log arrays and lists. If the list contains numerical values, it will also create a line plot of the values on the webpage.

```python
my_list = [1, 2, 3, 4, 5]

run["new_list"] = my_list
```
#### Training loop example

Logging as arrays/list is often used for logging of losses during the training/validation/testing.

```python
# Training loop
for epoch in range(10):
    # Some model
    loss = epoch**2 #Example loss

    run["training/loss"].append(loss) # Takes run["training/loss"] as an array
```

### Files

In case you need to upload some file such as an image, or .csv file.

```python
run["segmentation_example"].upload("segmentation_picture.png")
```

## Stopping the run

When you're done with the run, you can stop it by using:

```python
run.stop()
```

If you do not stop run and you just initialize new, nothing terrible usually happens - neptune will just assume it's still running for some time. **Although, be careful to not overwrite previous run! It's still better to stop the run manually.**

## Neptune.AI dashboard

After experimantation, you can see and create visualizations on the Neptune.AI website's dashboard. You can also see the data in tabular form, or create charts with multiple variables in it (which is useful for train/val loss chart).
