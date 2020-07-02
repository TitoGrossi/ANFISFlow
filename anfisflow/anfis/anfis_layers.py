'''
This files defines the classes for all necessary layers of the ANFIS model
'''

import tensorflow as tf
import itertools
import numpy as np


class FuzzyficationLayer(tf.keras.layers.Layer):
    '''
    This layer indicates the 1st layer of the model, which is defined by the
    fuzzyfication processes
    '''
    def __init__(self, num_inputs, num_mf, **kwargs):
        self.num_inputs = num_inputs
        self.num_mf = num_mf
        self.output_dim = num_mf*num_inputs
        super(FuzzyficationLayer, self).__init__(dynamic=True, **kwargs)

    def compute_output_shape(self, input_dim):
        return (None, 1, self.output_dim)

    def build(self, input_shape):
        '''
        This layer must have a certain number of built in parameters for each
        membership function of the inputs of the model. This number derives from the
        mf used, and it should be implemented for each subclass
        '''
        super(FuzzyficationLayer, self).build(input_shape)

    def call(self, x):
        '''output = []
        for k in range(len(x)):
            offset = 0
            o = []
            for j in range(self.num_inputs):
                for i in range(self.num_mf):
                    o.append(self.membership_function(x[k][j], *self.parameters[offset + i]))
                offset += self.num_mf
            output.append(o)
        return output'''
        output = []
        unstack = tf.unstack(x)
        for k in unstack:
            offset = 0
            o = []
            for j in range(self.num_inputs):
                for i in range(self.num_mf):
                    o.append(self.membership_function(k[j], *self.parameters[offset + i]))
                offset += self.num_mf
            output.append(o)
        return output

    def membership_function(self, x, *args):
        print('\nFuzzyficationLayer subclasses must implement the membership_function method\n')
        raise NotImplementedError


class FuzzyficationLayerGaussian(FuzzyficationLayer):
    def __init__(self, num_inputs, num_mf, **kwargs):
        super(FuzzyficationLayerGaussian, self).__init__(num_inputs, num_mf, **kwargs)

    def build(self, input_shape):
        self.parameters = self.add_weight(shape=(self.num_inputs*self.num_mf, 2), trainable=True, initializer='RandomNormal')
        super(FuzzyficationLayerGaussian, self).build(input_shape)

    @tf.function
    def membership_function(self, x, *args):
        return tf.math.exp((-(x - args[0])**2)/2*args[1]**2)


class FuzzyficationLayerBell(FuzzyficationLayer):
    def __init__(self, num_inputs, num_mf, **kwargs):
        super(FuzzyficationLayerBell, self).__init__(num_inputs, num_mf, **kwargs)

    def build(self, input_shape):
        self.parameters = self.add_weight(shape=(self.num_inputs*self.num_mf, 3), trainable=True, initializer='RandomNormal')
        super(FuzzyficationLayerBell, self).build(input_shape)

    @tf.function
    def membership_function(self, x, *args):
        return 1/(1 + tf.math.pow(tf.math.abs((x - args[2])/args[0]), 2*args[1]))


class FuzzyficationLayerTriangular(FuzzyficationLayer):
    def __init__(self, num_inputs, num_mf, **kwargs):
        super(FuzzyficationLayerTriangular, self).__init__(num_inputs, num_mf, **kwargs)

    def build(self, input_shape):
        self.parameters = self.add_weight(shape=(self.num_inputs*self.num_mf, 3), trainable=True, initializer='RandomNormal')
        super(FuzzyficationLayerTriangular, self).build(input_shape)

    @tf.function
    def membership_function(self, x, *args):
        return tf.mat.maximum(tf.math.minimum((x - args[0])/(args[1] - args[0]), (args[2] - x)/(args[2] - args[1])), 0)


class TNorm(tf.keras.layers.Layer):
    '''
    This layer represents the 2nd layer of the model, which is defined by the
    application of the T-norm in the fuzzyfied values
    '''
    def __init__(self, num_inputs, num_mf, **kwargs):
        self.output_dim = num_mf**num_inputs
        self.num_inputs = num_inputs
        self.num_mf = num_mf
        super(TNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        This layer has no parameters buit in them
        '''
        super(TNorm, self).build(input_shape)

    def call(self, x):
        '''
        Calling this layer must provide a differentiable set of operations, so
        we use the multiply AND operator
        '''
        # Multiply the membership value for each possible combination of inputs
        # e.g.: input1 is high (x[0]) and input2 is near (x[2])
        output = []
        for k in range(len(x)):
            divided_inputs = []
            for i in range(self.num_inputs):
                divided_inputs.append(x[k][i*self.num_mf:(i+1)*self.num_mf])
            comb = itertools.product(*divided_inputs)
            o = []
            for i in comb:
                mult_ = 1
                for j in i:
                    mult_ *= j
                o.append(mult_)
            output.append(o)
        return output


class NormFiringStrength(tf.keras.layers.Layer):
    '''
    This layer represents the 3rd layer of the model, which is defined by the
    normalization of the firing strengths to each rule (how strong it is in
    comparison to the others)
    '''
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NormFiringStrength, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        This layer has no parameters buit in them
        '''
        super(NormFiringStrength, self).build(input_shape)

    def call(self, x):
        '''
        To normalize the values, we just devide each of them by the sum of all
        of them
        '''
        sum = tf.keras.backend.sum(x) # Sum of inputs
        output = x/sum
        return output


class ConsequentRules(tf.keras.layers.Layer):
    def __init__(self, num_inputs, num_mf, **kwargs):
        self.num_inputs = num_inputs
        self.num_mf = num_mf
        self.output_dim = num_mf**num_inputs
        super(ConsequentRules, self).__init__(**kwargs)

    def build(self, input_dim):
        '''
        This layer has a set of parameters similar to the ones in dense layers.
        In fact, the first part of the node operations in this layer could be
        replaced with a built in Dense layer of Keras. They are represented in
        the model by an actual dense layer native to keras
        '''
        super(ConsequentRules, self).build(input_dim)
        #self.w = self.add_weight(shape=(self.num_inputs, self.output_dim), trainable=True, initializer='RandomNormal')
        #self.bias = self.add_weight(shape=(1,self.output_dim), trainable=True, initializer='Zeros')

    def call(self, x):
        # Separate the two inputs of this layer
        original_inputs, cons_rules_inputs = x
        # Dense operations
        #original_inputs = tf.matmul(original_inputs, self.w) + self.bias
        output = []
        for j in range(len(x[0])):
            o = []
            # Multiply result of dense with the output of layer 3
            for i in range(self.num_mf**self.num_inputs):
                o.append(original_inputs[j][i]*cons_rules_inputs[j][i])
            output.append(o)
        #output = tf.matmul(output, original_inputs)
        return output


class DeffuzzyficationLayer(tf.keras.layers.Layer):
    '''
    This layer represents the 5th layer of the model, which is defined by the
    deffuzyfication of the fuzzyfied values, turning them back to cripy values
    '''
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DeffuzzyficationLayer, self).__init__(**kwargs)

    def build(self, input_dim):
        '''
        This layer has no built in parameters
        '''
        super(DeffuzzyficationLayer, self).build(input_dim)

    def call(self, x):
        '''
        As applied in the ANFIS paper of Jyh-Shing Roger Jang, the output of the
        fifth layer of the type-3 Takagi-Sugeno type is the sum of the inputs
        '''
        return tf.keras.backend.sum(x)
