# Training Directory

This directory contains Jupyter notebooks and configuration files for training the internal and external classifiers used in the DATG (Dynamic Attribute Graphs-based Text Generation) framework. These classifiers are crucial for guiding the text generation process and for evaluating the success rate of sentiment transformation tasks.

## Notebooks

- `sentiment_classifier.ipynb`: Trains an internal classifier based on the BAAI/bge-large-en-v1.5 model using the IMDB dataset. This classifier is used within the DATG framework to guide the generation towards specific sentiment attributes.

- `toxic_classifier.ipynb`: Trains another internal classifier on the BAAI/bge-large-en-v1.5 model using the Jigsaw Toxic Comment dataset. This classifier is utilized to ensure the generated text avoids toxic content.

- `sentiment_discriminator.ipynb`: Develops an external classifier based on the FacebookAI/roberta-base model with the SST5 dataset. This is used for assessing the success rate of sentiment transformation tasks, serving as an evaluator outside the generation process.

## Configuration

- `train_config.py`: Provides configurations for fine-tuning the datasets and base models. Before running the notebooks, ensure this configuration file is properly set up with your desired parameters, including the paths to the datasets and the base model checkpoints.
