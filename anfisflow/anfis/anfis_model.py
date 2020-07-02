'''
This file contains the class for the ANFIS model itself, constructing the tensorflow
computational graph. The operations that happens inside the classes itself can be
checked in the ANFIS_Layers.py file
'''

import tensorflow as tf
from anfisflow.anfis import anfis_layers

class Model(tf.keras.models.Model):
    def __init__(self, num_inputs, num_mf, type_mf='gaussian'):
        # The eager execution is necessary for making iteration over tensor possible
        # although it is made to use for debugging
        tf.config.experimental_run_functions_eagerly(True)
        super(Model, self).__init__()
        if type_mf == 'gaussian':
            self.fuzzyfication = anfis_layers.FuzzyficationLayerGaussian(num_inputs, num_mf, input_shape=(None,num_inputs)) # 1st layer instance gaussian
        elif type_mf == 'bell':
            self.fuzzyfication = anfis_layers.FuzzyficationLayerBell(num_inputs, num_mf) # 1st layer instance bell
        elif type_mf == 'triangular':
            self.fuzzyfication = anfis_layers.FuzzyficationLayerTriangular(num_inputs, num_mf) # 1st layer instance triangular
        else:
            raise AssertionError
        self.t_norm = anfis_layers.TNorm(num_inputs, num_mf) # 2nd layer instance
        self.norm_fir_str = anfis_layers.NormFiringStrength(num_mf**num_inputs) # 3rd layer instance
        self.dense = tf.keras.layers.Dense(num_mf**num_inputs, trainable=True) # Dense layer representing 1st part of 4th layer operations
        self.conseq_rules = anfis_layers.ConsequentRules(num_inputs, num_mf) # 4th layer instance
        self.defuzz = anfis_layers.DeffuzzyficationLayer(1) # 5th layer instance

    def call(self, inputs):
        x = self.fuzzyfication(inputs)
        x = self.t_norm(x)
        x = self.norm_fir_str(x)
        d = self.dense(inputs)
        x = self.conseq_rules([d, x])
        y = self.defuzz(x)
        return y
