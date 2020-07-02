'''
Terminal tool for defining an ANFIS model and training it without getting to code.
This only allows a small set of control over the creation and fitting of the model,
compared to all the tools tensorflow and keras provides
'''

import argparse
import csv
import numpy as np
from tensorflow.keras import callbacks
import anfisflow.anfis.anfis_model

def read_input_data(input_file):
    '''
    read input file (csv type) and return the number of inputs for the model and
    the actual input and output data (the output column is suposed to be the last)
    column of the csv file!
    '''
    with open(input_file) as f:
        reader = csv.reader(f)
        input_data, output_data = [], []
        for row in reader:
            input_data.append([float(i) for i in row[:-1]])
            output_data.append(float(row[-1]))
        num_inputs = len(input_data[0])
    return num_inputs, np.asarray(input_data), np.asarray(output_data)

def train(num_inputs, num_mf, type_mf, epochs, input_data, input_target, optimizer, model_name, **kwargs):
    '''
    Train and save model with the terminal arguments passed
    '''
    model = anfisflow.anfis.anfis_model.Model(num_inputs, num_mf, type_mf=type_mf)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    callbacks_ = []
    validation_split = 0.0
    if 'tensorboard' in kwargs:
        logdir = kwargs.get('tensorboard')
        callbacks_.append(callbacks.TensorBoard(log_dir=logdir))
    if 'validation_split' in kwargs:
        validation_split = kwargs.get('validation_split')
    model.fit(input_data, input_target, epochs, validation_split=validation_split, callbacks=callbacks_)
    model.save_weights(model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build and train an ANFIS Model\
     based on Tekagi-Sugeno rules with the amount of inputs and membership functions\
     you choose, as long as the number of mfs for each input is the same')

    parser.add_argument('input_data', help='full path to csv file containing the\
                        input data of the model. The last column of the file will\
                        be considered the target and all the other ones features.\
                        Check for data inconsistencies before training!')

    parser.add_argument('num_mf', help='number of membership functions for the\
                        inputs of the model', type=int)

    parser.add_argument('type_mf', help='Type of mf used for the first layer', type=str)

    parser.add_argument('epochs', help='number of epochs for training', type=int)

    parser.add_argument('optimizer', help='optimizer used for training', type=str)

    parser.add_argument('output', help='name of model', type=str)

    parser.add_argument('--tensorboard', help='Do you want tensorboard as a callback during training?', type=str)

    parser.add_argument('--validation_split', help='Fraction of data used for validation', type=float)

    args = parser.parse_args()

    kwargs = {}
    if args.tensorboard:
        kwargs['tensorboard'] = args.tensorboard
    if args.validation_split:
        kwargs['validation_split'] = args.validation_split

    num_inputs, input_data, output_data = read_input_data(input_file=args.input_data)

    train(num_inputs, args.num_mf, args.type_mf, args.epochs, input_data, output_data, args.optimizer, args.output, **kwargs)
