# Fashion MNIST Classification Using Machine Learning Algorithms and CNN


## Introduction

Image classification has continued to rise in prominence as we continue to improve on the models we use. Within this domain, object recognition and classification have taken center stage, due to its multitude of use cases.  The Fashion-MNIST dataset, derived from the articles of Zalando’s catalog of clothes, has proved itself to be a more than adequate benchmark in order to evaluate the image classification capabilities of models.This dataset consists of a training set of 60,000 examples and a training set of 10,000 examples. Each example consists of a 28x28 grayscale image associated with one of 10 labels that correlate to a certain type of clothing such as bags, ankle boots, dressers, etc. The data is distributed evenly between all of the  labels, giving us 7,000 items for each label in total. As a result, we have 70,000 rows overall, each with 785 columns, which provides us with enough data to train a strong model. We will be using this dataset to train and validate multiple learning algorithms such as random forests classifier, support vector machines, extreme gradient boosting and convolutional neural networks in order to verify which of these is best suited to image classification tasks. 

## Dataset Structure

The Fashion-MNIST dataset is split into two subsets:

- Training Set: This subset contains 60,000 images used for training machine learning models.
- Testing Set: This subset consists of 10,000 images used for testing and benchmarking the trained models.

## Labels

Each training and test example is assigned to one of the following labels:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot
