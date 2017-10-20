# cifar_nn.py
# CIFAR 10 with Neural Networks in CNTK
#
# Things are purposely set up in a not-so-optimal way. Play with the network architecture (num hidden layers, their dimension, ...) 
# and the training hyperparameters (learning rate, minibatch size, number of epochs to train for, etc.) to try to improve the test error.

import cntk
import os
import sys

# define the data dimensions
input_dim = 3072
num_output_classes = 10

# data locations
data_dir = os.path.join('data', 'CIFAR')
train_file = os.path.join(data_dir, "Train_cntk_text.txt")
test_file = os.path.join(data_dir, "Test_cntk_text.txt")

# set up the CTF data reader
def create_reader(path, is_training, input_dim, num_label_classes):
    return cntk.io.MinibatchSource(cntk.io.CTFDeserializer(path, cntk.io.StreamDefs(
        labels    = cntk.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
        features  = cntk.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
    )), randomize = is_training, max_sweeps = cntk.io.INFINITELY_REPEAT if is_training else 1)
    

# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))
        
    return mb, training_loss, eval_error

def create_model(features, num_hidden_layers, hidden_layers_dim):
    with cntk.layers.default_options(init = cntk.layers.glorot_uniform(), activation = cntk.ops.relu):
        h = features
        for _ in range(num_hidden_layers):
            h = cntk.layers.Dense(hidden_layers_dim)(h)
        r = cntk.layers.Dense(num_output_classes, activation = None)(h)
        return r

def main():

    # model creation...
    num_hidden_layers = 1
    hidden_layers_dim = 64

    inputs = cntk.input_variable(input_dim)
    labels = cntk.input_variable(num_output_classes)

    # Normalize the features and create the model
    z = create_model(inputs / 255.0, num_hidden_layers, hidden_layers_dim)
    
    # training...
    loss = cntk.cross_entropy_with_softmax(z, labels)
    error = cntk.classification_error(z, labels)

    # Instantiate the trainer object to drive the model training
    learning_rate = 1e-10
    lr_schedule = cntk.learning_rate_schedule(learning_rate, cntk.UnitType.sample)
    learner = cntk.sgd(z.parameters, lr_schedule)
    trainer = cntk.Trainer(z, (loss, error), [learner])

    # Initialize the parameters for the trainer
    minibatch_size = 64
    epoch_size = 50000   # because there are 50000 training samples
    num_epochs = 5
    num_minibatches_to_train = (epoch_size * num_epochs) / minibatch_size

    # Create the reader to training data set
    reader_train = create_reader(train_file, True, input_dim, num_output_classes)

    # Map the data streams to the input and labels.
    input_map = {
        labels  : reader_train.streams.labels,
        inputs  : reader_train.streams.features
    } 

    # Run the trainer on and perform model training
    training_progress_output_freq = 500

    plotdata = {"batchsize":[], "loss":[], "error":[]}

    for i in range(0, int(num_minibatches_to_train)):
        # Read a mini batch from the training data file
        data = reader_train.next_minibatch(minibatch_size, input_map = input_map)
    
        trainer.train_minibatch(data)
        batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)    


    # eval / test...
    reader_test = create_reader(test_file, False, input_dim, num_output_classes)

    test_input_map = {
        labels : reader_test.streams.labels,
        inputs : reader_test.streams.features,
    }

    # Test data for trained model
    test_minibatch_size = 512
    num_samples = 10000
    num_minibatches_to_test = num_samples // test_minibatch_size
    test_result = 0.0

    for i in range(num_minibatches_to_test):
        data = reader_test.next_minibatch(test_minibatch_size, input_map = test_input_map)

        eval_error = trainer.test_minibatch(data)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))

if __name__ == '__main__':
    main()