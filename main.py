import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time

# Constants
MODEL_PATH = './models'
DATA_PATH = './data'
RUNS_PATH = './runs'
TB_PATH = './tb'
NUM_CLASSES = 2
EPOCHS = 2 #10, 15!, 1                                    # Number of epochs
BATCH_SIZE = 5 #5!,10,15                              # Reduce this depending on amount of RAM available
LEAR_RATE = 0.0001 #0.0005, 0.0001!, 0.001                # Initial learning rate
KEEP = 0.5 #0.8, 0.6, 0.5!                            # Probability keeping weights adjustment


def save_model(sess, epoch, dir_id_str):
    '''Save TensorFlow model variables to disk, the current epoch becomes part of the name'''
    saver = tf.train.Saver()
    save_path = saver.save(sess, MODEL_PATH +dir_id_str+'_epochs.ckpt')
    print("Model saved in file: %s" % save_path)


#def load_model(sess, epoch, dir_id_str):
#    '''Load previously saved TensorFlow model variables of a particular epoch'''
#    saver = tf.train.Saver()
#    saver.restore(sess, MODEL_PATH +dir_id_str+'epochs.ckpt')


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    input_image = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    vgg_layer3_out = graph.get_tensor_by_name('layer3_out:0')
    vgg_layer4_out = graph.get_tensor_by_name('layer4_out:0')
    vgg_layer7_out = graph.get_tensor_by_name('layer7_out:0')

    return input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

print('\nTesting load_vgg()...')
tests.test_load_vgg(load_vgg, tf)


def layer_1x1_conv(layer, num_classes, layer_name):
    return tf.layers.conv2d(layer, num_classes, 1,
                            strides=(1, 1),
                            name=layer_name+'_1x1_conv',
                            padding='same',
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))


def layer_transp(layer, num_classes, layer_name, kernel=4, strides=(2, 2), padding='same'):
    return tf.layers.conv2d_transpose(layer, num_classes, kernel,
                                      strides = strides,
                                      name = layer_name + '_transp_conv',
                                      padding = padding,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # 1x1 convolution: Use 1x1 convolution on layer 7 reducing # of classes to num_classes
    vgg_layer7_1x1 = layer_1x1_conv(vgg_layer7_out, num_classes, 'vgg_layer7')

    # Deconvolution: Upsample 2x using transposed convolution
    trans1 = layer_transp(vgg_layer7_1x1, num_classes, 'vgg_layer7')

    # Rescale VGG layer 4 (max pool) for compatibility as a skip layer
    scale_pool4 = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')

    # 1x1 convolution: Use 1x1 convolution on layer 4 reducing # of classes to num_classes
    scale_pool4 = layer_1x1_conv(scale_pool4, num_classes, 'vgg_layer4')

    #Skip: Build skip connection from layer 4 to layer 7
    skip1 = tf.add(trans1, scale_pool4)

    # Deconvolution: Upsample 2x using transposed convolution
    trans2 = layer_transp(skip1, num_classes, 'vgg_layer4')

    # Rescale VGG layer 3 (max pool) for compatibility as a skip layer
    scale_pool3 = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')

    # 1x1 convolution: Use 1x1 convolution on layer 3 reducing # of classes to num_classes
    scale_pool3 = layer_1x1_conv(scale_pool3, num_classes, 'vgg_layer3')

    # Skip: Build skip connection from layer 3 layer 4
    skip2 = tf.add(trans2, scale_pool3)

    # Deconvolution: Upsample 8x by transposed convolution
    final = layer_transp(skip2, num_classes, 'output', kernel=16, strides=(8, 8))

    return final

print('\nTesting layers()...')
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    cross_entropy_loss = tf.reduce_mean(cross_entropies)

    l2_reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.reduce_sum(l2_reg_losses)

    total_loss = cross_entropy_loss + regularization_loss

    # Create optimizer and training operation to minimize total loss
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(total_loss, global_step=global_step, name='train_op')

    # Compute accuracy for convenience
    prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    # Collecting summaries
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)

    return logits, train_op, total_loss, accuracy

print('\nTesting optimize()...')
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, accuracy, input_image,
             correct_label, keep_prob, learning_rate, dir_id_str):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input image
    :param correct_label: TF Placeholder for label image
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # Set up TensorBoard logging output
    tb_out_dir = TB_PATH+dir_id_str
    print("dir_runs_out =", tb_out_dir)
    tb_merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(tb_out_dir, sess.graph) # with graph
    #train_writer = tf.summary.FileWriter(tb_out_dir)  # without graph

    sess.run(tf.global_variables_initializer())

    image_batches = []
    for i in range(epochs):
        batch = 0
        print('Epoch %d' % (i))
        for image, label in get_batches_fn(batch_size):
            image_batches.append((image, label))
            batch += 1

            feed_dict = {input_image: image,
                         correct_label: label,
                         learning_rate: LEAR_RATE,
                         keep_prob: KEEP}

            loss, _, summary, batch_accuracy = sess.run([cross_entropy_loss, train_op, tb_merged, accuracy],
                                       feed_dict=feed_dict)

            print ('Batch %3d  total_loss %.03f  accuracy %.03f' % (batch, loss, batch_accuracy))

            # Log loss for each global step
            step = tf.train.global_step(sess, tf.train.get_global_step())
            train_writer.add_summary(summary,step)

        print("tensorboard --logdir {}".format(tb_out_dir))
        save_model(sess, i, dir_id_str)


def run():
    image_shape = (160, 576)
    print("Load data...")
    tests.test_for_kitti_dataset(DATA_PATH)

    # Download pretrained vgg model
    print("Loading VGG...")
    helper.maybe_download_pretrained_vgg(DATA_PATH)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_PATH, 'vgg')

        # Set up function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_PATH, 'data_road/training'), image_shape)

        # 1. Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, NUM_CLASSES)
        learning_rate = tf.placeholder(tf.float32, name = 'learning-rate')

        correct_label = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], NUM_CLASSES),
                                       name='correct-label')

        logits, train_op, cross_entropy_loss, accuracy = optimize(last_layer, correct_label, learning_rate, NUM_CLASSES)

        # Set up directory ID string to ref runs output with tensorboard inputs
        dir_id_str = "/"+str(time.time())+"_BS%d_DP%.02f_LR%.04f_EP%d" % (BATCH_SIZE, KEEP, LEAR_RATE, EPOCHS)

        # 2. Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op,
                 cross_entropy_loss, accuracy, input_image,
                 correct_label, keep_prob, learning_rate, dir_id_str)

        # 3. Save inference data using helper.save_inference_samples
        runs_dir = RUNS_PATH+dir_id_str
        helper.save_inference_samples(runs_dir, DATA_PATH, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    print("Starting main...")
    run()
