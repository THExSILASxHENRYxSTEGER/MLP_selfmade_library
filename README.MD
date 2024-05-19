The fashion_mnist_images folder contains the Fashion Mnist Data set of labeled hand drawn clothing pieces

The Data_prepare_MNIST.py file takes 300 and 100 labeled images from the training sets and testing sets in the fashion_mnist_images folder, respectively.

These images are then transformed into vectors and saved in Fashion_train.csv and Fashion_test.csv.

To change the number of parameters, i.e. number of images and where they are located from you can adjust the Data_prepare_MNIST.py file and execute it with the following command:

python3 Data_prepare_MNIST.py

However, as the neural networks in this library run on a cpu and not on a gpu the preset parameters are quite good to showcase the functionality of the library in a short amount of time.

The FASHION_MNIST_example executable file is unsurprisingly the executable of the FASHION_MNIST_example.cpp file.

The FASHION_MNIST_example.cpp file assembles in its main function a MLP of two dense layers with a ReLu activation function and a Softmax activation, respectively.

The loss function is Crossentropy and the optimizer is Adam.

All these layers and activations are Classes that are importet from the NN_LIB.h file which is a library in style smilar to pyTorch but only with the functionalities to build MLPs of any size and type.

to train the network only go into the folder on your command line and enter the command:

./FASHION_MNIST_example