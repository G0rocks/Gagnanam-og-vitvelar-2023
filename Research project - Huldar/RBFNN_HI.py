# Author:           Huldar
# Date:             2023-10-10
# Project:          Research project
# Acknowledgements: 
'''
RBFNN_HI is a module and an acronym which stands for
Radial Basis Function Neural Network Huldar Implementation
This module is my implementation of a radial basis function neural network. From now on radial basis function neural network will be abbreviated as RBFNN.

Mostly this module allows you to create an instance of the RBFNN class and use the class methods on the class to create your own RBFNN.
'''

import numpy    # Used to work with arrays

class RBFNN:
    '''
    RBFNN class.
    '''

    def __init__(self, n_nodes: int=10) -> None:
        '''
        Initialization function for the RBFNN class.

        Inputs:
        n_nodes:    (int) number of nodes in the hidden layer of the RBFNN, defaults to 10

        Outputs:
        There are no outputs
        '''
        self.nodes = numpy.zeros(n_nodes)


    def train(self, train_features, train_targets):
        '''
        Train RBFNN

        Inputs:
        train_features: Dictionary where the keys are the         
        '''

        pass

    def test(self, test_features, test_targets):
        pass