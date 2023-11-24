import numpy 
from random import uniform, randint

class Leaky_Relu():
    def __init__(self, before_point_gradient=0, after_point_gradient=1, change_point=0):
        self.before_point_gradient = before_point_gradient
        self.after_point_gradient = after_point_gradient
        self.change_point = change_point 

    def derivative(self, X):
        return [
            (self.before_point_gradient if x < self.change_point else self.after_point_gradient) 
            for x in X
        ]

    def standard(self, X):
        return [
            (self.before_point_gradient * x if x < self.change_point else self.after_point_gradient * x)
            for x in X
        ]

class MSE():
    def derivative(self, Y, true_Y):
        R = (Y-true_Y)
        n = len(Y)

        # not not sure how picking R vs R.T affects dimensions in calculations
        return 2/n * R
    
    def standard(self, Y, true_Y):
        R = (Y-true_Y)
        n = len(Y)        
        # return 1/n * R * R.T
        return 1/n * numpy.dot(R, R)

        

class FFN():
    def __init__(self, neurons_per_layer: tuple[int], encoder=None, decoder=None, activation_function=None, activate_last_layer=False) -> None:
        if encoder is None: encoder = lambda X:X
        if decoder is None: decoder = lambda X:X
        if activation_function is None: activation_function = Leaky_Relu()

        self.encoder = encoder
        self.decoder = decoder

        self.LR = 10^-2
        self.LR_increase = 1.05
        self.LR_decrease = 1-(self.LR_increase)**1/5
        self.num_transformation_layers = len(neurons_per_layer)-1
        self.num_neuron_activation_layers = len(neurons_per_layer)

        self.AF = activation_function.standard
        self.dAF_dx = activation_function.derivative  
        self.activate_last_layer = activate_last_layer

        self.dimensions = {
            "neuron_activation_vector": neurons_per_layer,
            "weight_matrix": [None for _ in range(self.num_transformation_layers)],
            "bias_vector": [None for _ in range(self.num_transformation_layers)],
        }

        # organised by transformation and then by weight or bias
        self.trainable_parameters = [{"B": None, "W": None} for _ in range(self.num_transformation_layers)]

        # iterate thrugh transformations
        for transormation_index in range(self.num_transformation_layers):
            # caluclate parameter dimensions
            self.dimensions["bias_vector"][transormation_index] = self.dimensions["neuron_activation_vector"][transormation_index+1]
            self.dimensions["weight_matrix"][transormation_index] = [
                self.dimensions["neuron_activation_vector"][transormation_index+1],
                self.dimensions["neuron_activation_vector"][transormation_index]
            ]

            # create inital values for parameters
            self.trainable_parameters[transormation_index]["B"] = numpy.array([0,]*self.dimensions["bias_vector"][transormation_index])

            starting_weight_random_range = 1
            self.trainable_parameters[transormation_index]["W"] = numpy.array([
                [
                    uniform(-starting_weight_random_range, starting_weight_random_range)
                    for _ in range(self.dimensions["weight_matrix"][transormation_index][1])
                ]
                for _ in range(self.dimensions["weight_matrix"][transormation_index][0])
            ])

    def predict(self, X):
        predict_verbose = False

        if predict_verbose: print(f"Making prediction with input:   {X}")
        X = self.encoder(X)
        if predict_verbose: print(f"Making prediction with encoded input vector:   {[round(x, 2) for x in X]}")

        neuron_activation_vector = numpy.array(X)
        for transformation_index in range(self.num_transformation_layers):
            parameters = self.trainable_parameters[transformation_index]

            if predict_verbose: print(f"transformation {transformation_index+1} transforming  {[round(x, 2) for x in neuron_activation_vector]}")
            if predict_verbose: print(f"Weighted Sum is   {[round(x, 2) for x in (parameters['W'] @ neuron_activation_vector)]}")
            if predict_verbose: print(f"Adding bias gives is   {[round(x, 2) for x in ((parameters['W'] @ neuron_activation_vector) + parameters['B'])]}")

            if self.activate_last_layer or (transformation_index != self.num_transformation_layers-1): 
                if predict_verbose: print(f"putting though activation function gives gives is   {[round(x, 2) for x in self.AF((parameters['W'] @ neuron_activation_vector) + parameters['B'])]}")

            neuron_activation_vector = ((parameters["W"] @ neuron_activation_vector) + parameters["B"])
            if self.activate_last_layer or (transformation_index != self.num_transformation_layers-1):
                neuron_activation_vector = self.AF(neuron_activation_vector)  

        if predict_verbose: print(f"Output vector decoded is   {self.decoder(neuron_activation_vector)}")

        return self.decoder(neuron_activation_vector)
    
    def compute_atomic_derivative():
        pass
    
    def backpropigate_to_find_compound_derivative():
        pass

    def train(self, X_data, Y_data, minibatch_size, epochs):
        pass


def scalar_vector_encoding_factory(min, max, neurons):
    def bounded_value(x):
        if x <= 0: return 0.0
        if x >= 1: return 1.0
        else: return x

    def encoder(x):
        # convert x to range [0, neurons]
        coded_x = (x - min) / (max - min) * neurons
        X = [bounded_value(coded_x-i) for i in range(neurons)]
        return X

    def decoder(X):
        coded_x = sum(bounded_value(x) for x in X)
        x = (coded_x * (max - min) / neurons) + min
        return x
    return encoder, decoder