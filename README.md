# Fashion MNIST Classification Using Machine Learning Algorithms and CNN


## Introduction

Image classification has continued to rise in prominence as we continue to improve on the models we use. Within this domain, object recognition and classification have taken center stage, due to its multitude of use cases.  The Fashion-MNIST dataset, derived from the articles of Zalando’s catalog of clothes, has proved itself to be a more than adequate benchmark in order to evaluate the image classification capabilities of models.This dataset consists of a training set of 60,000 examples and a training set of 10,000 examples. Each example consists of a 28x28 grayscale image associated with one of 10 labels that correlate to a certain type of clothing such as bags, ankle boots, dressers, etc. The data is distributed evenly between all of the  labels, giving us 7,000 items for each label in total. As a result, we have 70,000 rows overall, each with 785 columns, which provides us with enough data to train a strong model. We will be using this dataset to train and validate multiple learning algorithms such as random forests classifier, support vector machines, extreme gradient boosting and convolutional neural networks in order to verify which of these is best suited to image classification tasks. 

## Dataset Structure

The Fashion-MNIST dataset is split into two subsets:

- Training Set: This subset contains 60,000 images used for training machine learning models.
- Testing Set: This subset consists of 10,000 images used for testing and benchmarking the trained models.

[Click here to download the dataset.](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

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

# Sample Images with Labels

<img width="417" alt="sample_images_w_labels" src="https://github.com/archit28-tamu/fashion_mnist_project/assets/143445547/b26ad3d0-af58-4b47-8337-87a9747056cc">


## Data Preprocessing

Starting with the distribution of the data, we observe that data is uniformly distributed with each class having 6000 images in the training dataset.

<img width="647" alt="class_distribution" src="https://github.com/archit28-tamu/fashion_mnist_project/assets/143445547/1e604dec-b521-4cb4-86f8-8e8289da873e">


In order to streamline the workflow, we will be reducing the dimensionality of data. For this we will be using principal component analysis. After plotting a graph between explained variance and the number of components, we can see that we retain about 95% variance at 150 components.

As such, we will keep 150 principal components. The more principal components we use, the higher the separability of the data ought to be, and this is showcased in the graphs below; where we can clearly see that the separability between the different classes markedly increases as we increase the components from 2 to 3.

For CNN, we will split the training data into training and validation sets, with the validation set having 10,000 rows. We will also range normalize the pixel data. This is done by dividing the pixel data by 255, as all of the pixel values lie between 0 and 255. The data is also reshaped into its intended 28x28 format so that the model reads the data as images, rather than a set of values.

## Model Selection and Evaluation

For model selection, we used various machine learning models like SVM, Random Forest, Logistic Regression and XGBoost.

For feature extraction, Principle Component Analysis was done with 2 componentes. Feature extraction was done to extract the most important features from the dataset and droping the less important features. Reducing the number of features help in speeding up the training of the models where we are trainig large dataset. The advantage we get from PCA is that we get good computation speed without losing out much on accuracy.

Experiments were done on these models including hyperparameter-tuning to get accuracies of the models to choose the best performing model with best set of hyperparameters. Following are the accuracies of various machine learning models that we have used:

| **Model**          | **Accuracy**|
|:-------------------|------------:|
| SVM                | 90.32%      |
| Random Forest      | 86.98%      |
| Logistic Regression| 85.16%      |
| XGBoost            | 89.05%      |

After machine learning models, we also experiment with deep learning. For working with images, Convulational Neural Network produce very good results, so we try building architecture for CNN models. We built 3 different architectures of CNN with different combination of Convulation Layers, Batch Normalization, Max Pooling and Dropout. We optimized every architecture based on the Class Activation Map that we generated from convulation layers. Also, we tunned hyperparameter by changing the number of epochs and finally trainind our last model with 25 epochs.

So, finally, the best accuracy we are getting with our experiment with different machine learning models and different architectures of CNN is with this architecture of the model:

<img width="389" alt="cnn3_architecture" src="https://github.com/archit28-tamu/fashion_mnist_project/assets/143445547/a165ddb2-a434-4661-9712-5c415cef14be">


And the test accuracy of the model we achieved is 93.1%.

<img width="539" alt="cnn3_accuracy" src="https://github.com/archit28-tamu/fashion_mnist_project/assets/143445547/0a20cd26-17dd-4d7c-a141-3206b93742f6">


**Loss Evaluation and Accuracy Evaluation Charts**


<img width="728" alt="loss_accuracy_evaluation" src="https://github.com/archit28-tamu/fashion_mnist_project/assets/143445547/62e715c0-586a-4710-bedb-78976ec3afeb">


## Interpretability

Convolutional neural networks assign importance to various aspects in an image using weights and biases and use these aspects in order to differentiate between images. This is clearly visible when we visualize the class activation maps. We can see from the class activation maps of the best performing architecture given below.

![Class Activation Maps for CNN Architecture](https://github.com/archit28-tamu/fashion_mnist_project/assets/143130477/b8b98b25-d708-4b5a-a29d-75724a58c077)


We can see in the first activation map, the first convolutional layer activates the pixels of the entire sneaker while activating the heel and the topline of the sneaker. This indicates to us that the model will be considering the general shape and size of the sneaker, while prioritizing the existence/shape of the heel and topline of the sneaker. From the second activation map we see that the second convolutional layer has prioritized only the laces and the tongue of the sneaker along with parts of the sole, barely activating anything else; indicating to us that the model will be paying attention to the laces and tongue of the sneaker in the second layer. As we reach the third layer, the class activation map starts to become more abstract. Whereas the previous two maps resembled the original image quite well, the third image is starting to deviate from the original quite heavily. However, it is not degraded enough to become a detriment, as we can make out that the third layer has earmarked the tongue and the topline of the sneaker.

## Results

Multiple machine learning algorithms such as logistic regression, support vector machines, random forests classifier, extreme gradient boost and convolutional neural networks in order to find out which of these models perform the best on the Fashion-MNIST dataset. The convolutional neural network performed the best with a 93.1% accuracy.

## Conclusion

In this project, we have conducted a series of tests in an effort to find the best model to classify the Fashion-MNIST dataset, and as an extension identify a model that will be proficient at identifying and classifying clothing images. We conducted a preliminary data exploration in order to get an idea of what we will be working with. The data is then transformed using principal component analysis and normalization in order to improve the efficiency of the model.  The data is then divided into three sections: train, test and validate sets so as to train and validate the model without cross-contaminating the test set. Using this newly processed data, we train several machine learning models; namely, logistic regression, random forests classifier, support vector machine, extreme gradient boost and convolutional neural network. Our results demonstrate that convolutional neural networks outperform other machine learning algorithms for image classification tasks, achieving an accuracy of 93%. This remarkable accuracy highlights the effectiveness of convolutional neural networks in image classification tasks and paves the way for further advancements in this domain.

## References

- https://www.kaggle.com/code/eliotbarr/fashion-mnist-tutorial
- https://www.kaggle.com/code/faressayah/fashion-mnist-classification-using-cnns
- https://www.tensorflow.org/tutorials/images/cnn
- https://www.pinecone.io/learn/class-activation-maps/
- https://www.kaggle.com/datasets/zalando-research/fashionmnist
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://xgboost.readthedocs.io/en/stable/


## License

MIT License

Copyright (c) 2023 archit28-tamu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
