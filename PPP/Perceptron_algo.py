import numpy as np
import pandas as pd
import os

fileDir = os.path.dirname(os.path.abspath(__file__))
learning_rate= 0.01
epochs= 500
testSeeds= fileDir+'\\testSeeds.csv'
trainSeeds= fileDir+'\\trainSeeds.csv'
txt = fileDir+'\\result.txt'
text_file = open(txt,'w')

def output_function(activation):
    
    # !decision of the threshold value is made by the first activation array value
    if activation > 0.5:
        return 1
    return 0

def error_calculation(output, actual_output):

    '''
    output -> output based on the calculation
    actual_output -> desired output
    '''
    if output > actual_output:
        #prediction is greater than actual
        return -1
    
    if output < actual_output:
        #prediction is smaller than actual
        return 1
    
    # being equal
    return 0

def update_w(pre_w, learning_r, delta_w, col_num):

    '''
    pre_w -> previous weight (original weight)
    delta_w -> +1/-1 * input depends on the value of calculation
    using [0][x],[1][x] to update each column independently.
    '''
    pre_w[0][col_num] = pre_w[0][col_num] + delta_w[0] * learning_r
    pre_w[1][col_num] = pre_w[1][col_num] + delta_w[1] * learning_r

    # bias update
    pre_w[2][col_num] = pre_w[2][col_num] + delta_w[2] * learning_r
    return pre_w

def accuracy(output_arr,act_out):

    ## failed prediction
    if ((output_arr == [0, 0, 0]).all()):
        return 0

    ## successful prediction
    out = np.argmax(output_arr, axis=0) + 1 #+1 cuz argmax returns the index
    print('estimate out is ',out, ', and it is', (act_out == out).all() )

    #######################################
    #log the prediction and actual result
    #######################################
    text_file.write('\n result is: %d actual output is: %d'%(out,act_out.values[0]))


    # prediction is right
    if ((act_out == out).all()):
        return 1
    
    # prediction is wrong
    return 0

def pred_test(weights,path):
    df_test = pd.read_csv(path,header=None)
    X_test = df_test.iloc[:,3:5]
    X_test['bias'] = 1

    y_test = df_test.iloc[:,-1]
    outputs= [0,0,0]
    correctness = 0
    for i in range(0,X_test.shape[0]):

        inputs_test= X_test.iloc[i : (i+1)]
        activation_test= np.dot(inputs_test, weights)

        outputs= np.array([0,0,0])

        for idx in range(0,activation_test.shape[1]):
            outputs[idx]= output_function(activation_test[0,idx])

        actual_output= y_test.iloc[i : (i+1)]
        print('\n result is:',outputs, 'actual output:', actual_output.values[0])

        correctness += accuracy(outputs, actual_output)

    print('classification error is %d out of %d' %((X_test.shape[0] - correctness),(X_test.shape[0])))

def examine(actual_out, output, inputs,weights):
    for index,element in enumerate(actual_out):
        error= error_calculation(output[index],element)
        delta_weight = np.dot(error, inputs.T)
        weights = update_w(weights, learning_rate, delta_weight, index)


if __name__ == "__main__":

    df = pd.read_csv(trainSeeds ,header=None)

    # Using two features: length & width
    X_train = df.iloc[:,3:5]
    X_train['bias'] = 1

    y_train = df.iloc[:,-1]

    random = np.random.RandomState(1)

    # num of input neurons
    n_inputs= 2

    # num of output neurons
    n_outputs= 3

    #######################################
    # Weight and bias Initialization
    #######################################

    weights = random.normal(loc=0.0,scale=0.1,size=(n_inputs + 1, n_outputs))
    print('initial weights: \n', weights)

    ########################################
    # Training Part
    ########################################

    for k in range(0, epochs):
        print(k, 'epoch starts')

        # looping through every data sample.
        for i in range(0,X_train.shape[0]):

            #print(i,' sample update')
            inputs= X_train.iloc[i : (i+1)]
            activations= np.dot(inputs, weights)
            outputs= np.array([0,0,0])

            # storing the result into an array
            # Example activation: [0.233, -0.01, -0.6] => output: [1, 0, 0]
            for idx in range(0,activations.shape[1]):
                outputs[idx]= output_function(activations[0,idx])
            
            # actual_output can be either 1, 2, 3
            actual_output= y_train.iloc[i : (i+1)]

            if (actual_output == 1).all():
                #expecting [1, 0, 0]
                examine([1,0,0], outputs, inputs, weights)

            if (actual_output == 2).all():
                #expecting [0, 1, 0]
                examine([0,1,0], outputs, inputs, weights)

            if (actual_output == 3).all():
                #expecting [0, 0, 1]
                examine([0,0,1], outputs, inputs, weights)
                
    print('final weights:\n', weights)

    ########################################
    # Predictions
    ########################################
    text_file.write('Below is testing set result:')
    pred_test(weights,testSeeds)
    
    # for spliting two training results
    print("##########################################################################")

    text_file.write('\n\nBelow is training set result:')
    pred_test(weights,trainSeeds)
    
    # Close Writing
    text_file.close()

