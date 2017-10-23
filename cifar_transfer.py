# cifar_transfer.py
#
# CIFAR 10 with Convolutional Neural Networks in CNTK, using transfer learning.
#
# We will use transfer learning to learn a new model based on a pre-trained ResNet.
# I mentioned ResNet-34 in the assignment, but we'll actually use ResNet-20.
# Download 'ResNet20.model' from the Sharepoint.
#
# You will have to fill in the function create_model(). This involves first finding the name
# of the final pooling layer that you can find by plotting the loaded-up base model. You will 
# load the model, then call cntk.logging.graph.plot(model, 'output_name.pdf'). Take a look at the 
# PDF and find the name of the node that feeds into the final dense layer (the last weight and bias parameters).
# Then, you'll need to get a reference to that node using the depth_first_search() function (as outlined below).
# Use that to create a new CNTK function (using the combined() function) and then calling clone() on it with the 
# appropriate clone method (see the CNTK documentation at www.cntk.ai). Finally, you will run your input_features 
# through that cloned network and add the output of that into a Dense layer like you've done in other CNTK networks.

import numpy as np
import sys
import os
import cntk

# data locations
data_dir   = os.path.join('data', 'CIFAR')
train_file = os.path.join(data_dir, "Train_cntk_text.txt")
test_file  = os.path.join(data_dir, "Test_cntk_text.txt")
base_model_file = "ResNet20.model"

# set up the CTF data reader
def create_reader(path, is_training, input_dim, num_label_classes):
    return cntk.io.MinibatchSource(cntk.io.CTFDeserializer(path, cntk.io.StreamDefs(
        labels    = cntk.io.StreamDef(field='labels',   shape=num_label_classes, is_sparse=False),
        features  = cntk.io.StreamDef(field='features', shape=input_dim,         is_sparse=False)
    )), randomize = is_training, max_sweeps = cntk.io.INFINITELY_REPEAT if is_training else 1)

# Creates the network model for transfer learning
def create_model(base_model_file, num_classes, input_features):

    # set the name of the final pooling layer
    last_pool = "BlockSomething"

    # Load the pretrained classification net and find the last node...

    # use the load_model function to load up the existing pre-trained ResNet model
    base_model = 

    # get a reference to the last node using the depth_first_search() function
    last_node  = cntk.logging.graph.depth_first_search(base_model, lambda n: n.uid.startswith(last_pool))[0]

    # clone all the layers up to the final pooling layer
    # use combine() on the last_node's owner to create a CNTK function, then use clone() with the appropriate CloneMethod
    cloned_layers = 

    # Add new dense layer with an output that matches our prediction space
    cloned_out = # run the input_features through the cloned_layers

    # now create a Dense layer and put the output of the pre-trained CNN (cloned_out) through it
    # with the number of output classes set to the size of our output space
    z = 

    return z

def main():

    # set the image input details and the size of the output space (our 10 CIFAR-10 categories)
    image_height = 32
    image_width  = 32
    num_channels = 3
    input_dim = image_height * image_width * num_channels
    num_output_classes = 10

    # input variables -- note that this time we keep the actual shape of the input images rather than flattening them
    input_var = cntk.ops.input_variable((num_channels, image_height, image_width))
    label_var = cntk.ops.input_variable(num_output_classes)

    # normalize the features  (or at least approximately normalize them)
    input_removemean = cntk.ops.minus(input_var, cntk.ops.constant(128))
    scaled_input = cntk.ops.element_times(cntk.ops.constant(0.004), input_removemean)

    z = create_model(base_model_file, num_output_classes, scaled_input)

    ce = cntk.losses.cross_entropy_with_softmax(z, label_var)
    pe = cntk.metrics.classification_error(z, label_var)

    reader_train = create_reader(train_file, True, input_dim, num_output_classes)

    # set learning parameters
    epoch_size = 50000
    minibatch_size = 64
    max_epochs = 30

    lr_per_sample = [0.001]*10 + [0.0005]*10 + [0.0001]
    lr_schedule   = cntk.learning_rate_schedule(lr_per_sample, cntk.UnitType.sample, epoch_size=epoch_size)
    mm            = 0.9
    mm_schedule   = cntk.learners.momentum_schedule(mm)
    l2_reg_weight = 0.002

    # Instantiate the trainer object to drive the model training
    learner = cntk.learners.momentum_sgd(z.parameters, lr_schedule, mm_schedule, l2_regularization_weight = l2_reg_weight)
    progress_printer = cntk.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
    trainer = cntk.Trainer(z, (ce, pe), learner, progress_printer)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var  : reader_train.streams.features,
        label_var  : reader_train.streams.labels
    }

    # print the number of parameters that are about to be learned
    cntk.logging.log_number_of_parameters(z) ; print()

    # Get minibatches of images to train with and perform model training
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it
            sample_count += trainer.previous_minibatch_sample_count         # count samples processed so far

        trainer.summarize_training_progress()
        z.save("CIFAR10_{}.dnn".format(epoch))

    # Load test data
    reader_test = create_reader(test_file, False, input_dim, num_output_classes)

    input_map = {
        input_var  : reader_test.streams.features,
        label_var  : reader_test.streams.labels
    }

    # Test data for trained model
    epoch_size = 10000
    minibatch_size = 16

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    sample_count    = 0
    minibatch_index = 0

    while sample_count < epoch_size:
        current_minibatch = min(minibatch_size, epoch_size - sample_count)
        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)
        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch
        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples
        minibatch_index += 1

    print("")
    print("Final Test Results: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")

    return metric_numer/metric_denom

if __name__ == '__main__':
    main()
