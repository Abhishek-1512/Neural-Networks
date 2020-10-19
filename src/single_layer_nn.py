# Jain, Abhishek
# 1001-759-977
# 2020-09-27
# Assignment-01-01

import numpy as np

class SingleLayerNN(object):
    def __init__(self, input_dimensions=2,number_of_nodes=4):

        self.input_dimensions=input_dimensions                   #setting the parameter input_dimensions equal to the number of dimensions of the input data
        self.number_of_nodes=number_of_nodes                    # setting the parameter number_of_nodes equal to the number of neurons in the model
        """
        Initialize SingleLayerNN model and set all the weights and biases to random numbers.
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: Note that number of neurons in the model is equal to the number of classes.
        """
        self.initialize_weights()                   # intializing the weights

    def initialize_weights(self,seed=None):
        
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:                      # using seed to initialize the weights if the seed is given
            np.random.seed(seed)

        self.weights=[]                                                                         
        self.weights=np.random.randn(self.number_of_nodes,self.input_dimensions+1)             #initialize the weights using random number
        return None

    def set_weights(self, W):
        """
        This function sets the weight matrix (Bias is included in the weight matrix).
        :param W: weight matrix
        :return: None if the input matrix, w, has the correct shape.
        If the weight matrix does not have the correct shape, this function
        should not change the weight matrix and it should return -1.
        """

        a=np.shape(W)
        if(self.number_of_nodes==a[0] and self.input_dimensions+1==a[1]):           #checking if the weight matrix has correct shape or not
            self.weights=W                                                        #setting the weight matrix
            return None
        else:                                                                       #returning -1 and not changing weight matrix if the shape is not correct
            return -1

    def get_weights(self):
        """
        This function should return the weight matrix(Bias is included in the weight matrix).
        :return: Weight matrix
        """
        return self.weights                                                     # returning the weight matrix

    def predict(self, X):
        """
        Make a prediction on a batach of inputs.
        :param X: Array of input [input_dimensions,n_samples]
        :return: Array of model [number_of_nodes ,n_samples]
        Note that the activation function of all the nodes is hard limit.
        """
        X=np.insert(X,0,1,axis=0)                          #converting array of input to matrix form
        net=np.dot(self.weights,X)                      #getting the net value or output by doing dot product of weight matrix and input matrix
        predicted_output=np.where(net[:] <0, 0,1)            #activation function  
        return predicted_output

    def train(self,X,Y,num_epochs=10,alpha=0.1):
        """
        Given a batch of input and desired outputs, and the necessary hyperparameters (num_epochs and alpha),
        this function adjusts the weights using Perceptron learning rule.
        Training should be repeated num_epochs times.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        X = np.insert(X,0,1,axis=0)
        for x in range(num_epochs):                                             #running the training function epoch number of times
            for y in range(len(X[0])):                                          #running for all the inputs
                trained_input=np.expand_dims(X[:,y],axis =1)                  #comparing input with the target
                net=np.dot(self.weights,trained_input)                        #getting the net value
                target=np.expand_dims(Y[:,y],axis=1) 
                prediction=np.where(net[:] <0, 0,1)                           #applying the activation function
                error=target-prediction                                     #calculating error by subtracting the new values from the target value
                error_percentage=error.dot(np.transpose(trained_input))       #applying error to inputs to get error percentage   
                self.weights=self.weights+alpha*(error_percentage)          #updating weights
        return None

    def calculate_percent_error(self,X,Y):
        """
        Given a batch of input and desired outputs, this function calculates percent error.
        For each input sample, if the output is not the same as the desired output, Y,
        then it is considered one error. Percent error is 100*(number_of_errors/ number_of_samples).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :return percent_error
        """
        #calculating percent error using the given formula 
        #Percent error = 100*(number_of_errors/ number_of_samples)

        z=0
        predicted_output=self.predict(X)                               
        for x in range(len(X[0])):
            prediction=np.expand_dims(predicted_output[:,x],axis=1)
            target=np.expand_dims(Y[:,x],axis=1)
            if(np.array_equal(prediction,target)):
                z+=0
            else:
                z+=1
            j=z/len(X[0])*100
        return j


if __name__ == "__main__":
    input_dimensions = 2
    number_of_nodes = 2

    model = SingleLayerNN(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=2)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    print("****** Model weights ******\n",model.get_weights())
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.1)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.get_weights())