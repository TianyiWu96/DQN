"""Thin wrapper around TensorFlow logic."""
import tensorflow as tf


class QNet(object):
    """A deep network Q-approximator implemented with TensorFlow.

    The network is structure is fixed, aside from the output width, which depends
    on the number of actions necessary to play a given game.
    """

    def __init__(self, output_width, learning_rate):
        """Initializes the TensorFlow graph.

        Args:
            output_width: The number of output units.
        """
        self.graph_in, self.graph_out = self.construct_graph(output_width)

        self.target_reward = tf.placeholder(tf.float32)
        self.action_idxs = tf.placeholder(tf.int32)
        actual_reward = tf.gather_nd(self.graph_out, self.action_idxs)
        self.loss = tf.reduce_mean(
            tf.square(self.target_reward - actual_reward))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def __del__(self):
        """Closes the TensorFlow session, freeing resources."""
        self.sess.close()

    def compute_q(self, net_in):
        """Forward-propagates the given input and returns the array of outputs.

        Args:
            net_in: Image to forward-prop through the network. Must be 1x84x84x4.

        Returns:
            The array of network outputs.
        """
        return self.sess.run(self.graph_out, feed_dict={self.graph_in:[net_in]})[0]

    def update(self, batch_frames, batch_actions, batch_targets):
        """Updates the network with the given batch input/target values using RMSProp.

        Args:
            batch_frames: Set of Nx84x84x4 network inputs, where N is the batch size.
            batch_actions: Set of N action indices, representing action taken at each state.
            batch_target: Corresponding target Q values for each input.
        """
        # Note: Action indicies must actually be tuples since graph_out is a 2D tensor.
        # To prevent tight coupling, modify the batch_actions list here.
        return self.sess.run(
            [self.optimizer, self.loss],
            feed_dict={
                self.graph_in:batch_frames,
                self.action_idxs:[tup for tup in enumerate(batch_actions)],
                self.target_reward:batch_targets})[1]
    
    def construct_graph(self, output_width, frame_height=84, frame_width=84, state_frames=4, use_pooling=False, use_dueling=False):
        """Creates a new TensorFlow graph with predetermined structure.

        Args:
            output_width: The number of output units for the graph.

        Returns:
            The graph input and output tensors (in that order).
        """
        graph_in = tf.placeholder(
            tf.float32,
            shape=[None, frame_height, frame_width, state_frames])

        w_conv1 = self._weight_variable([8, 8, state_frames, 32])
        b_conv1 = self._bias_variable([32])
        w_conv2 = self._weight_variable([4, 4, 32, 64],)
        b_conv2 = self._bias_variable([64])
        w_conv3 = self._weight_variable([3, 3, 64, 64])
        b_conv3 = self._bias_variable([64])
        
        if use_pooling:
            conv_layer1 = tf.nn.relu(_conv2d(graph_in, w_conv1, 4) + b_conv1)
            pool_layer1 = self._pool(conv_layer1)

            conv_layer2 = tf.nn.relu(_conv2d(pool_layer1, w_conv2, 2) + b_conv2)
            pool_layer2 = self._pool(conv_layer2)

            conv_layer3 = tf.nn.relu(_conv2d(pool_layer2, w_conv3, 1) + b_conv3)
            pool_layer3 = self._pool(conv_layer3)

            conv_layer3_flat = tf.reshape(pool_layer3, [-1, 256])
            w_fc1 = self._weight_variable([256, 256])
            b_fc1 = self._bias_variable([256])
            fc_layer1 = tf.nn.relu(tf.matmul(conv_layer3_flat, w_fc1) + b_fc1)

            w_fc2 = self._weight_variable([256, output_width])
            b_fc2 = self._bias_variable([output_width])
            graph_out = tf.add(tf.matmul(fc_layer1, w_fc2), b_fc2)

        else:
            conv_layer1 = tf.nn.relu(self._conv2d(graph_in, w_conv1, 4) + b_conv1)

            conv_layer2 = tf.nn.relu(self._conv2d(conv_layer1, w_conv2, 2) + b_conv2)

            conv_layer3 = tf.nn.relu(self._conv2d(conv_layer2, w_conv3, 1) + b_conv3)

            # TODO: Reshape this so we don't have to hardcode the the number of inputs
            # and can freely change the frame height/width.
            conv_layer3_flat = tf.reshape(conv_layer3, [-1, 7744])

            # Convolutional layer 3 to fully connected layer 1
            w_fc1 = self._weight_variable([7744, 512])
            b_fc1 = self._bias_variable([512])
            fc_layer1 = tf.nn.relu(tf.matmul(conv_layer3_flat, w_fc1) + b_fc1)

            # Bias and weights for fully connected layer 1
            w_fc2 = self._weight_variable([512, output_width])
            b_fc2 = self._bias_variable([output_width])
            graph_out = tf.add(tf.matmul(fc_layer1, w_fc2), b_fc2)

        return graph_in, graph_out


    def _conv2d(self, data, weights, stride):
        """Returns a TensforFlow 2D convolutional layer.

        Args:
            data: The input tensor to the convolutional layer.
            weights: The convolutional weights for this layer.
            stride: The x and y stride for the convolution.

        Returns:
            The TensorFlow convolutional layer.
        """
        return tf.nn.conv2d(data, weights, strides=[1, stride, stride, 1], padding='SAME')

    def _pool(self, data, stride=2):
        """Returns a TensforFlow pooling layer.
        Args:
            data: The input tensor to the pooling layer.
        Returns:
            The TensorFlow pooling layer.
        """
        return tf.nn.max_pool(data, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')


    def _weight_variable(self, shape):
        """Returns a TensforFlow weight variable.

        Args:
            shape: The shape of the weight variable.

        Returns:
            A TensorFlow weight variable of the given size.
        """
        return tf.Variable(tf.truncated_normal(shape, stddev=0.01))


    def _bias_variable(self, shape):
        """Returns a TensforFlow 2D bias variable.

        Args:
            shape: The shape of the bias variable.

        Returns:
            A TensorFlow bias variable of the specified shape.
        """
        return tf.Variable(tf.constant(0.01, shape=shape))

