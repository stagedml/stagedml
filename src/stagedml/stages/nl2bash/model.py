
import tensorflow as tf
import numpy as np

from six.moves import xrange
from typing import Any, Tuple, Optional

PAD_ID=0 # FIXME: copied from data_utils


class NNModel(object):
    def __init__(self, hyperparams, buckets=None)->None:
        self.hyperparams = hyperparams
        self.buckets = buckets

    # --- model architecture hyperparameters --- #

    @property
    def encoder_topology(self):
        return self.hyperparams["encoder_topology"]

    @property
    def decoder_topology(self):
        return self.hyperparams["decoder_topology"]

    @property
    def num_layers(self):
        return self.hyperparams["num_layers"]

    # --- training algorithm hyperparameters --- #

    @property
    def training_algorithm(self):
        return self.hyperparams["training_algorithm"]

    @property
    def use_sampled_softmax(self):
        return self.num_samples > 0 and \
               self.num_samples < self.target_vocab_size

    @property
    def num_samples(self):
        return self.hyperparams["num_samples"]

    @property
    def batch_size(self):
        return self.hyperparams["batch_size"]

    @property
    def num_epochs(self):
        return self.hyperparams["num_epochs"]

    @property
    def steps_per_epoch(self):
        return self.hyperparams["steps_per_epoch"]

    @property
    def max_gradient_norm(self):
        return self.hyperparams["max_gradient_norm"]

    @property
    def optimizer(self):
        return self.hyperparams["optimizer"]

    @property
    def margin(self):
        return self.hyperparams["margin"]

    @property
    def adam_epsilon(self):
        return self.hyperparams["adam_epsilon"]

    @property
    def tg_token_use_attention(self):
        return self.hyperparams["tg_token_use_attention"]

    @property
    def tg_token_attn_fun(self):
        return self.hyperparams["tg_token_attn_fun"]

    @property
    def variational_recurrent_dropout(self):
        return self.hyperparams["variational_recurrent_dropout"]

    @property
    def attention_input_keep(self):
        return self.hyperparams["attention_input_keep"]

    @property
    def attention_output_keep(self):
        return self.hyperparams["attention_output_keep"]

    @property
    def rnn_cell(self):
        return self.hyperparams["rnn_cell"]

    @property
    def gamma_c(self):
        return self.hyperparams["gamma_c"]

    @property
    def beta_c(self):
        return self.hyperparams["beta_c"]

    @property
    def gamma_h(self):
        return self.hyperparams["gamma_h"]

    @property
    def beta_h(self):
        return self.hyperparams["beta_h"]

    @property
    def gamma_x(self):
        return self.hyperparams["gamma_x"]

    @property
    def beta_x(self):
        return self.hyperparams["beta_x"]


    @property
    def source_vocab_size(self):
        return self.hyperparams["source_vocab_size"]

    @property
    def target_vocab_size(self):
        return self.hyperparams["target_vocab_size"]

    @property
    def max_source_length(self):
        return self.hyperparams["max_source_length"]

    @property
    def max_target_length(self):
        return self.hyperparams["max_target_length"]

    @property
    def max_source_token_size(self):
        return self.hyperparams["max_source_token_size"]

    @property
    def max_target_token_size(self):
        return self.hyperparams["max_target_token_size"]

    @property
    def decode_sig(self):
        return self.hyperparams["decode_sig"]

    @property
    def model_dir(self):
        return self.hyperparams["model_dir"]

    @property
    def sc_token(self):
        return self.hyperparams["sc_token"]

    @property
    def sc_token_dim(self):
        """
        Source token channel embedding dimension.
        """
        return self.hyperparams["sc_token_dim"]

    @property
    def sc_input_keep(self):
        return self.hyperparams["sc_input_keep"]

    @property
    def sc_output_keep(self):
        return self.hyperparams["sc_output_keep"]

    @property
    def sc_token_features_path(self):
        return self.hyperparams["sc_token_features_path"]

    @property
    def sc_char(self):
        return self.hyperparams["sc_char"]

    @property
    def sc_char_vocab_size(self):
        return self.hyperparams["sc_char_vocab_size"]

    @property
    def sc_char_dim(self):
        """
        Source character channel embedding dimension.
        """
        return self.hyperparams["sc_char_dim"]

    @property
    def sc_char_composition(self):
        return self.hyperparams["sc_char_composition"]

    @property
    def sc_char_rnn_cell(self):
        return self.hyperparams["sc_char_rnn_cell"]

    @property
    def sc_char_rnn_num_layers(self):
        return self.hyperparams["sc_char_rnn_num_layers"]

    @property
    def sc_char_features_path(self):
        return self.hyperparams["sc_char_features_path"]

    @property
    def tg_input_keep(self):
        return self.hyperparams["tg_input_keep"]

    @property
    def tg_output_keep(self):
        return self.hyperparams["tg_output_keep"]

    @property
    def tg_token_features_path(self):
        return self.hyperparams["tg_token_features_path"]

    @property
    def tg_char(self):
        return self.hyperparams["tg_char"]

    @property
    def tg_char_vocab_size(self):
        return self.hyperparams["tg_char_vocab_size"]

    @property
    def tg_char_composition(self):
        return self.hyperparams["tg_char_composition"]

    @property
    def tg_char_rnn_cell(self):
        return self.hyperparams["tg_char_rnn_cell"]

    @property
    def tg_char_use_attention(self):
        return self.hyperparams["tg_char_use_attention"]

    @property
    def tg_char_rnn_num_layers(self):
        return self.hyperparams["tg_char_rnn_num_layers"]

    @property
    def tg_char_features_path(self):
        return self.hyperparams["tg_char_features_path"]

    @property
    def tg_char_rnn_input_keep(self):
        return self.hyperparams["tg_char_rnn_input_keep"]

    @property
    def tg_char_rnn_output_keep(self):
        return self.hyperparams["tg_char_rnn_output_keep"]

    @property
    def gamma(self):
        return self.hyperparams["gamma"]

    # -- copy mechanism -- #

    @property
    def use_copy(self):
        return self.hyperparams["use_copy"]

    @property
    def copy_fun(self):
        return self.hyperparams["copy_fun"]

    @property
    def copynet(self):
        return self.use_copy and self.copy_fun == 'copynet'

    @property
    def copy_vocab_size(self):
        return self.hyperparams["copy_vocab_size"]

    @property
    def chi(self):
        return self.hyperparams["chi"]

    # --- decoding algorithm hyperparameters --- #

    @property
    def forward_only(self):
        # If set, we do not construct the backward pass in the model.
        return self.hyperparams["forward_only"]

    @property
    def token_decoding_algorithm(self):
        return self.hyperparams["token_decoding_algorithm"]

    @property
    def char_decoding_algorithm(self):
        return self.hyperparams["char_decoding_algorithm"]

    @property
    def beam_size(self):
        return self.hyperparams["beam_size"]

    @property
    def beam_order(self):
        return self.hyperparams["beam_order"]

    @property
    def alpha(self):
        return self.hyperparams["alpha"]

    @property
    def beta(self):
        return self.hyperparams["beta"]

    @property
    def top_k(self):
        return self.hyperparams["top_k"]

    @property
    def force_reading_input(self):
        return self.hyperparams["force_reading_input"]

def wrap_inputs(beam_decoder, inputs):
    return [beam_decoder.wrap_input(input) for input in inputs]


def sparse_cross_entropy(logits, targets):
    return -tf.reduce_sum(input_tensor=logits * tf.one_hot(targets, logits.get_shape()[1]), axis=1)


def nest_map(func, nested):
    """
    Apply function to each element in a nested list.

    :param func: The function to apply.
    :param nested: The nested list to which the function is going to be applied.

    :return: A list with the same structue as nested where the each element
        is the output of applying func to the corresponding element in nest.
    """
    if not nest.is_sequence(nested):
        return func(nested)
    flat = nest.flatten(nested)
    return nest.pack_sequence_as(nested, list(map(func, flat)))


def nest_map_dual(func, nested1, nested2):
    if not nest.is_sequence(nested1):
        return func(nested1, nested2)
    flat1 = nest.flatten(nested1)
    flat2 = nest.flatten(nested2)
    output = [func(x, y) for x, y in zip(flat1, flat2)]
    return nest.pack_sequence_as(nested1, list(output))


def create_multilayer_cell(rnn_cell, scope, dim, num_layers, input_keep_prob=1,
                           output_keep_prob=1, variational_recurrent=True, input_dim=-1):
    """
    Create the multi-layer RNN cell.
    :param type: Type of RNN cell.
    :param scope: Variable scope.
    :param dim: Dimension of hidden layers.
    :param num_layers: Number of layers of cells.
    :param input_keep_prob: Proportion of input to keep in dropout.
    :param output_keep_prob: Proportion of output to keep in dropout.
    :param variational_recurrent: If set, use variational recurrent dropout.
        (cf. https://arxiv.org/abs/1512.05287)
    :param input_dim: RNN input dimension, must be specified if it is
        different from the cell state dimension.
    :param batch_normalization: If set, use recurrent batch normalization.
        (cf. https://arxiv.org/abs/1603.09025)
    :param forward_only: If batch_normalization is set, inform the cell about
        the batch normalization process.
    :return: RNN cell as specified.
    """
    with tf.compat.v1.variable_scope(scope):
        if rnn_cell == "lstm":
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(dim, state_is_tuple=True)
        elif rnn_cell == "gru":
            cell = tf.compat.v1.nn.rnn_cell.GRUCell(dim)
        else:
            raise ValueError("Unrecognized RNN cell type: {}.".format(type))

        assert(input_keep_prob >= 0 and output_keep_prob >= 0)
        if input_keep_prob < 1 or output_keep_prob < 1:
            if input_dim == -1:
                input_dim = dim
            print("-- rnn dropout input keep probability: {}".format(input_keep_prob))
            print("-- rnn dropout output keep probability: {}".format(output_keep_prob))
            if variational_recurrent:
                print("-- using variational dropout")
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell,
                input_keep_prob=input_keep_prob,
                output_keep_prob=output_keep_prob,
                variational_recurrent=variational_recurrent,
                input_size=input_dim, dtype=tf.float32)

        if num_layers > 1:
            cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [cell] * num_layers, state_is_tuple=(rnn_cell=="lstm"))
    return cell

def softmax_loss(output_project, num_samples, target_vocab_size):
    w, b = output_project
    if num_samples > 0 and num_samples < target_vocab_size:
        print("loss function = sampled_softmax_loss ({})".format(num_samples))
        w_t = tf.transpose(a=w)
        def sampled_loss(outputs, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(
                w_t, b, labels, outputs, num_samples, target_vocab_size)
        loss_function = sampled_loss
    else:
        print("loss function = softmax_loss")
        def loss(outputs, labels):
            logits = tf.matmul(outputs, w) + b
            return tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
        loss_function = loss
    return loss_function




class Example(object):
    """
    Input data to the neural network (batched when mini-batch training is used).
    """
    def __init__(self)->None:
        self.encoder_inputs:Optional[list] = None
        self.encoder_attn_masks:Optional[list] = None
        self.decoder_inputs:Optional[list] = None
        self.target_weights:Optional[list] = None
        self.encoder_copy_inputs:Optional[list] = None     # Copynet
        self.copy_targets:Optional[list] = None            # Copynet


class Output(object):
    """
    Data output from the neural network (batched when mini-batch training is used).
    """
    def __init__(self)->None:
        self.updates = None
        self.gradient_norms = None
        self.losses = None
        self.output_symbols = None
        self.sequence_logits = None
        self.attn_alignments = None
        self.encoder_hidden_states = None
        self.decoder_hidden_states = None
        self.pointers = None



















class EncoderDecoderModel(NNModel):

    def __init__(self, hyperparams, buckets=None)->None:
        """Create the model.
        Hyperparameters:
          source_vocab_size: size of the source vocabulary.
          target_vocab_size: size of the target vocabulary.
          buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].e
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          use_attention: if set, use attention model.
        """
        super(EncoderDecoderModel, self).__init__(hyperparams, buckets)
        self.learning_rate = tf.Variable(
            float(hyperparams["learning_rate"]), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * hyperparams["learning_rate_decay_factor"])

        self.global_epoch = tf.Variable(0, trainable=False)

        # Encoder.
        self.define_encoder(self.sc_input_keep, self.sc_output_keep)

        # Decoder.
        decoder_embedding_dim = self.encoder.output_dim
        decoder_dim = decoder_embedding_dim
        self.define_decoder(decoder_dim, decoder_embedding_dim,
                            self.tg_token_use_attention,
                            self.tg_token_attn_fun,
                            self.tg_input_keep,
                            self.tg_output_keep)

        # Character Decoder.
        if self.tg_char:
            assert False, 'broken'
            self.define_char_decoder(self.decoder.dim, False,
                self.tg_char_rnn_input_keep, self.tg_char_rnn_output_keep)

        self.define_graph()

    # --- Graph Operations --- #

    def define_graph(self)->None:
        self.debug_vars:list = []

        # Feeds for inputs.
        self.encoder_inputs = []        # encoder inputs.
        self.encoder_attn_masks = []    # mask out PAD symbols in the encoder
        self.decoder_inputs = []        # decoder inputs (always start with "_GO").
        self.targets = []               # decoder targets
        self.target_weights = []        # weights at each position of the target sequence.
        self.encoder_copy_inputs = []

        for i in xrange(self.max_source_length):
            self.encoder_inputs.append(
                tf.compat.v1.placeholder(
                    tf.int32, shape=[None], name="encoder{0}".format(i)))
            self.encoder_attn_masks.append(
                tf.compat.v1.placeholder(
                    tf.float32, shape=[None], name="attn_alignment{0}".format(i)))

        for j in xrange(self.max_target_length + 1):
            self.decoder_inputs.append(
                tf.compat.v1.placeholder(
                    tf.int32, shape=[None], name="decoder{0}".format(j)))
            self.target_weights.append(
                tf.compat.v1.placeholder(
                    tf.float32, shape=[None], name="weight{0}".format(j)))
            # Our targets are decoder inputs shifted by one.
            if j > 0 and not self.copynet:
                self.targets.append(self.decoder_inputs[j])

        if self.copynet:
            for i in xrange(self.max_source_length):
                self.encoder_copy_inputs.append(
                    tf.compat.v1.placeholder(
                        tf.int32, shape=[None], name="encoder_copy{0}".format(i)))
            for j in xrange(self.max_target_length):
                self.targets.append(
                    tf.compat.v1.placeholder(
                        tf.int32, shape=[None], name="copy_target{0}".format(i)))

        # Compute training outputs and losses in the forward direction.
        if self.buckets:
            self.output_symbols = []
            self.sequence_logits = []
            self.losses = []
            self.attn_alignments = []
            self.encoder_hidden_states = []
            self.decoder_hidden_states = []
            if self.tg_char:
                self.char_output_symbols = []
                self.char_sequence_logits = []
            if self.use_copy:
                self.pointers = []
            for bucket_id, bucket in enumerate(self.buckets):
                with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(),
                                       reuse=True if bucket_id > 0 else None):
                    print("creating bucket {} ({}, {})...".format(
                            bucket_id, bucket[0], bucket[1]))
                    encode_decode_outputs = \
                        self.encode_decode(
                            [self.encoder_inputs[:bucket[0]]],
                            self.encoder_attn_masks[:bucket[0]],
                            self.decoder_inputs[:bucket[1]],
                            self.targets[:bucket[1]],
                            self.target_weights[:bucket[1]],
                            encoder_copy_inputs=self.encoder_copy_inputs[:bucket[0]]
                        )
                    self.output_symbols.append(encode_decode_outputs['output_symbols'])
                    self.sequence_logits.append(encode_decode_outputs['sequence_logits'])
                    self.losses.append(encode_decode_outputs['losses'])
                    self.attn_alignments.append(encode_decode_outputs['attn_alignments'])
                    self.encoder_hidden_states.append(
                        encode_decode_outputs['encoder_hidden_states'])
                    self.decoder_hidden_states.append(
                        encode_decode_outputs['decoder_hidden_states'])
                    if self.forward_only and self.tg_char:
                         bucket_char_output_symbols = \
                             encode_decode_outputs['char_output_symbols']
                         bucket_char_sequence_logits =  \
                             encode_decode_outputs['char_sequence_logits']
                         self.char_output_symbols.append(
                             tf.reshape(bucket_char_output_symbols,
                                        [self.max_target_length,
                                         self.batch_size, self.beam_size,
                                         self.max_target_token_size + 1]))
                         self.char_sequence_logits.append(
                             tf.reshape(bucket_char_sequence_logits,
                                        [self.max_target_length,
                                        self.batch_size, self.beam_size]))
                    if self.use_copy:
                        self.pointers.append(encode_decode_outputs['pointers'])
        else:
            encode_decode_outputs = self.encode_decode(
                [self.encoder_inputs],
                self.encoder_attn_masks,
                self.decoder_inputs,
                self.targets,
                self.target_weights,
                encoder_copy_inputs=self.encoder_copy_inputs
            )
            self.output_symbols = encode_decode_outputs['output_symbols']
            self.sequence_logits = encode_decode_outputs['sequence_logits']
            self.losses = encode_decode_outputs['losses']
            self.attn_alignments = encode_decode_outputs['attn_alignments']
            self.encoder_hidden_states = encode_decode_outputs['encoder_hidden_states']
            self.decoder_hidden_states = encode_decode_outputs['decoder_hidden_states']
            if self.tg_char:
                char_output_symbols = encode_decode_outputs['char_output_symbols']
                char_sequence_logits = encode_decode_outputs['char_sequence_logits']
                self.char_output_symbols = tf.reshape(char_output_symbols,
                                   [self.batch_size, self.beam_size,
                                    self.max_target_length,
                                    self.max_target_token_size])
                self.char_sequence_logits = tf.reshape(char_sequence_logits,
                                   [self.batch_size, self.beam_size,
                                    self.max_target_length])
            if self.use_copy:
                self.pointers = encode_decode_outputs['pointers']

        # Gradients and SGD updates in the backward direction.
        if not self.forward_only:
            params = tf.compat.v1.trainable_variables()
            if self.optimizer == "sgd":
                opt = tf.compat.v1.train.GradientDescentOptimizer(self.learning_rate)
            elif self.optimizer == "adam":
                opt = tf.compat.v1.train.AdamOptimizer(
                    self.learning_rate, beta1=0.9, beta2=0.999,
                    epsilon=self.adam_epsilon, )
            else:
                raise ValueError("Unrecognized optimizer type.")

            if self.buckets:
                self.gradient_norms = []
                self.updates = []
                for bucket_id, _ in enumerate(self.buckets):
                    gradients = tf.gradients(ys=self.losses[bucket_id], xs=params)
                    clipped_gradients, norm = tf.clip_by_global_norm(
                        gradients, self.max_gradient_norm)
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(
                        zip(clipped_gradients, params)))
            else:
                gradients = tf.gradients(ys=self.losses, xs=params)
                clipped_gradients, norm = tf.clip_by_global_norm(
                    gradients, self.max_gradient_norm)
                self.gradient_norms = norm
                self.updates = opt.apply_gradients(zip(clipped_gradients, params))

        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())


    def encode_decode(self, encoder_channel_inputs, encoder_attn_masks,
                      decoder_inputs, targets, target_weights,
                      encoder_copy_inputs=None)->dict:
        bs_decoding = self.token_decoding_algorithm == 'beam_search' \
            and self.forward_only

        # --- Encode Step --- #
        if bs_decoding:
            targets = wrap_inputs(
                self.decoder.beam_decoder, targets)
            encoder_copy_inputs = wrap_inputs(
                self.decoder.beam_decoder, encoder_copy_inputs)
        encoder_outputs, encoder_states = \
            self.encoder.define_graph(encoder_channel_inputs)

        # --- Decode Step --- #
        if self.tg_token_use_attention:
            attention_states = tf.concat(
                [tf.reshape(m, [-1, 1, self.encoder.output_dim])
                 for m in encoder_outputs], axis=1)
        else:
            attention_states = None
        num_heads = 2 if (self.tg_token_use_attention and self.copynet) else 1

        output_symbols, sequence_logits, output_logits, states, attn_alignments, \
            pointers = self.decoder.define_graph(
                        encoder_states[-1], decoder_inputs,
                        encoder_attn_masks=encoder_attn_masks,
                        attention_states=attention_states,
                        num_heads=num_heads,
                        encoder_copy_inputs=encoder_copy_inputs)

        # --- Compute Losses --- #
        if not self.forward_only:
            # A. Sequence Loss
            if self.training_algorithm == "standard":
                encoder_decoder_token_loss = self.sequence_loss(
                    output_logits, targets, target_weights,
                    sparse_cross_entropy)
            elif self.training_algorithm == 'beam_search_opt':
                pass
            else:
                raise AttributeError("Unrecognized training algorithm.")

            # B. Attention Regularization
            attention_reg = self.attention_regularization(attn_alignments) \
                if self.tg_token_use_attention else 0

            # C. Character Sequence Loss
            if self.tg_char:
                # re-arrange character inputs
                char_decoder_inputs = [
                    tf.squeeze(x, 1) for x in tf.split(
                        axis=1, num_or_size_splits=self.max_target_token_size + 2,
                        value=tf.concat(axis=0, values=self.char_decoder_inputs))]
                char_targets = [
                    tf.squeeze(x, 1) for x in tf.split(
                        axis=1, num_or_size_splits=self.max_target_token_size + 1,
                        value=tf.concat(axis=0, values=self.char_targets))]
                char_target_weights = [
                    tf.squeeze(x, 1) for x in tf.split(
                        axis=1, num_or_size_splits=self.max_target_token_size + 1,
                        value=tf.concat(axis=0, values=self.char_target_weights))]
                if bs_decoding:
                    char_decoder_inputs = wrap_inputs(
                        self.decoder.beam_decoder, char_decoder_inputs)
                    char_targets = graph_utils.wrap_inputs(
                        self.decoder.beam_decoder, char_targets)
                    char_target_weights = graph_utils.wrap_inputs(
                        self.decoder.beam_decoder, char_target_weights)
                # get initial state from decoder output
                char_decoder_init_state = \
                    tf.concat(axis=0, values=[tf.reshape(d_o, [-1, self.decoder.dim])
                                              for d_o in states])
                char_output_symbols, char_sequence_logits, char_output_logits, _, _ = \
                    self.char_decoder.define_graph(
                        char_decoder_init_state, char_decoder_inputs)
                encoder_decoder_char_loss = self.sequence_loss(
                    char_output_logits, char_targets, char_target_weights,
                    softmax_loss(
                        self.char_decoder.output_project,
                        self.tg_char_vocab_size / 2,
                        self.tg_char_vocab_size))
            else:
                encoder_decoder_char_loss = 0

            losses = encoder_decoder_token_loss + \
                     self.gamma * encoder_decoder_char_loss + \
                     self.beta * attention_reg
        else:
            losses = tf.zeros_like(decoder_inputs[0])

        # --- Store encoder/decoder output states --- #
        encoder_hidden_states = tf.concat(
            axis=1, values=[tf.reshape(e_o, [-1, 1, self.encoder.output_dim])
                            for e_o in encoder_outputs])

        top_states = []
        if self.rnn_cell == 'gru':
            for state in states:
                top_states.append(state[:, -self.decoder.dim:])
        elif self.rnn_cell == 'lstm':
            for state in states:
                if self.num_layers > 1:
                    top_states.append(state[-1][1])
                else:
                    top_states.append(state[1])
        decoder_hidden_states = tf.concat(axis=1,
            values=[tf.reshape(d_o, [-1, 1, self.decoder.dim])
                    for d_o in top_states])

        O = {}
        O['output_symbols'] = output_symbols
        O['sequence_logits'] = sequence_logits
        O['losses'] = losses
        O['attn_alignments'] = attn_alignments
        O['encoder_hidden_states'] = encoder_hidden_states
        O['decoder_hidden_states'] = decoder_hidden_states
        if self.tg_char:
            O['char_output_symbols'] = char_output_symbols
            O['char_sequence_logits'] = char_sequence_logits
        if self.use_copy:
            O['pointers'] = pointers
        return O


    def sequence_loss(self, logits, targets, target_weights, loss_function):
        assert(len(logits) == len(targets))
        with tf.compat.v1.variable_scope("sequence_loss"):
            log_perp_list = []
            for logit, target, weight in zip(logits, targets, target_weights):
                crossent = loss_function(logit, target)
                log_perp_list.append(crossent * weight)
            log_perps = tf.add_n(log_perp_list)
            total_size = tf.add_n(target_weights)
            log_perps /= total_size

        avg_log_perps = tf.reduce_mean(input_tensor=log_perps)

        return avg_log_perps


    def attention_regularization(self, attn_alignments):
        """
        Entropy regularization term.
        :param attn_alignments: [batch_size, decoder_size, encoder_size]
        """
        P = tf.reduce_sum(input_tensor=attn_alignments, axis=1)
        P_exp = tf.exp(P)
        Z = tf.reduce_sum(input_tensor=P_exp, axis=1, keepdims=True)
        return tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=P_exp / Z * (P - tf.math.log(Z)), axis=1))


    def define_encoder(self, input_keep, output_keep):
        """Placeholder function."""
        self.encoder = None


    def define_decoder(self, dim, embedding_dim, use_attention,
                       attention_function, input_keep, output_keep):
        """Placeholder function."""
        self.decoder = None


    def define_char_decoder(self, dim, use_attention, input_keep, output_keep):
        """
        Define the decoder which does character-level generation of a token.
        """
        if self.tg_char_composition == 'rnn':
            self.char_decoder = rnn_decoder.RNNDecoder(self.hyperparams,
                "char_decoder", self.tg_char_vocab_size, dim, use_attention,
                input_keep, output_keep, self.char_decoding_algorithm)
        else:
            raise ValueError("Unrecognized target character composition: {}."
                             .format(self.tg_char_composition))

    # --- Graph Operations --- #

    def format_batch(self, encoder_input_channels, decoder_input_channels, bucket_id=-1)->Example:
        """
        Convert the feature vectors into the dimensions required by the neural
        network.
        :param encoder_input_channels:
            channel 0 - seq2seq encoder inputs
            channel 1 - copynet encoder copy inputs
        :param decoder_input_channels:
            channel 0 - seq2seq decoder inputs
            channel 1 - copynet decoder targets
        """
        def load_channel(inputs, output_length, reversed_output=True)->list:
            """
            Convert a batch of feature vectors into a batched feature vector.
            """
            padded_inputs = []
            batch_inputs = []
            for batch_idx in xrange(batch_size):
                input = inputs[batch_idx]
                paddings = [PAD_ID] * (output_length - len(input))
                if reversed_output:
                    padded_inputs.append(list(reversed(input + paddings)))
                else:
                    padded_inputs.append(input + paddings)
            for length_idx in xrange(output_length):
                batched_dim = np.array([padded_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(batch_size)], dtype=np.int32)
                batch_inputs.append(batched_dim)
            return batch_inputs

        if bucket_id != -1:
            encoder_size, decoder_size = self.buckets[bucket_id]
        else:
            encoder_size, decoder_size = \
                self.max_source_length, self.max_target_length
        batch_size = len(encoder_input_channels[0])

        # create batch-major vectors
        batch_encoder_inputs = load_channel(
            encoder_input_channels[0], encoder_size, reversed_output=True)
        batch_decoder_inputs = load_channel(
            decoder_input_channels[0], decoder_size, reversed_output=False)
        if self.copynet:
            batch_encoder_copy_inputs = load_channel(
                encoder_input_channels[1], encoder_size, reversed_output=True)
            batch_copy_targets = load_channel(
                decoder_input_channels[1], decoder_size, reversed_output=False)

        batch_encoder_input_masks = []
        batch_decoder_input_masks = []
        for length_idx in xrange(encoder_size):
            batch_encoder_input_mask = np.ones(batch_size, dtype=np.float32)
            for batch_idx in xrange(batch_size):
                source = batch_encoder_inputs[length_idx][batch_idx]
                if source == PAD_ID:
                    batch_encoder_input_mask[batch_idx] = 0.0
            batch_encoder_input_masks.append(batch_encoder_input_mask)

        for length_idx in xrange(decoder_size):
            # Create target_weights to be 0 for targets that are padding.
            batch_decoder_input_mask = np.ones(batch_size, dtype=np.float32)
            for batch_idx in xrange(batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = batch_decoder_inputs[length_idx+1][batch_idx]
                if length_idx == decoder_size - 1 or target == PAD_ID:
                    batch_decoder_input_mask[batch_idx] = 0.0
            batch_decoder_input_masks.append(batch_decoder_input_mask)

        E = Example()
        E.encoder_inputs = batch_encoder_inputs
        E.encoder_attn_masks = batch_encoder_input_masks
        E.decoder_inputs = batch_decoder_inputs
        E.target_weights = batch_decoder_input_masks
        if self.use_copy:
            E.encoder_copy_inputs = batch_encoder_copy_inputs
            E.copy_targets = batch_copy_targets

        return E


    def get_batch(self, data, bucket_id=-1, use_all=False)->Example:
        """
        Randomly sample a batch of examples from the specified bucket and
        convert the feature vectors into the dimensions required by the neural
        network.
        """
        encoder_inputs, decoder_inputs = [], []
        if self.copynet:
            encoder_copy_inputs, copy_targets = [], []

        if bucket_id == -1:
            sample_pool = data
        else:
            sample_pool = data[bucket_id]

        # Randomly sample a batch of encoder and decoder inputs from data
        data_ids = list(xrange(len(sample_pool)))
        if not use_all:
            data_ids = np.random.choice(data_ids, self.batch_size)
        for i in data_ids:
            example = sample_pool[i]
            encoder_inputs.append(example.sc_ids)
            decoder_inputs.append(example.tg_ids)
            if self.copynet:
                encoder_copy_inputs.append(example.csc_ids)
                copy_targets.append(example.ctg_ids)

        encoder_input_channels = [encoder_inputs]
        decoder_input_channels = [decoder_inputs]
        if self.copynet:
            encoder_input_channels.append(encoder_copy_inputs)
            decoder_input_channels.append(copy_targets)

        return self.format_batch(
            encoder_input_channels, decoder_input_channels, bucket_id=bucket_id)


    def feed_input(self, E)->dict:
        """
        Assign the data vectors to the corresponding neural network variables.
        """
        encoder_size, decoder_size = len(E.encoder_inputs), len(E.decoder_inputs)
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = E.encoder_inputs[l]
            input_feed[self.encoder_attn_masks[l].name] = E.encoder_attn_masks[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = E.decoder_inputs[l]
            input_feed[self.target_weights[l].name] = E.target_weights[l]
        if self.copynet:
            for l in xrange(encoder_size):
                input_feed[self.encoder_copy_inputs[l].name] = \
                    E.encoder_copy_inputs[l]
            for l in xrange(decoder_size-1):
                input_feed[self.targets[l].name] = E.copy_targets[l]

        # Apply dummy values to encoder and decoder inputs
        for l in xrange(encoder_size, self.max_source_length):
            input_feed[self.encoder_inputs[l].name] = np.zeros(
                E.encoder_inputs[-1].shape, dtype=np.int32)
            input_feed[self.encoder_attn_masks[l].name] = np.zeros(
                E.encoder_attn_masks[-1].shape, dtype=np.int32)
            if self.copynet:
                input_feed[self.encoder_copy_inputs[l].name] = \
                    np.zeros(E.encoder_copy_inputs[-1].shape, dtype=np.int32)
        for l in xrange(decoder_size, self.max_target_length + 1):
            input_feed[self.decoder_inputs[l].name] = np.zeros(
                E.decoder_inputs[-1].shape, dtype=np.int32)
            input_feed[self.target_weights[l].name] = np.zeros(
                E.target_weights[-1].shape, dtype=np.int32)
            if self.copynet:
                input_feed[self.targets[l-1].name] = np.zeros(
                    E.copy_targets[-1].shape, dtype=np.int32)

        return input_feed


    def step(self, session, formatted_example, bucket_id=-1, forward_only=False)->Output:
        """Run a step of the model feeding the given inputs.
        :param session: tensorflow session to use.
        :param encoder_inputs: list of numpy int vectors to feed as encoder inputs.
        :param attn_alignments: list of numpy int vectors to feed as the mask
            over inputs about which tokens to attend to.
        :param decoder_inputs: list of numpy int vectors to feed as decoder inputs.
        :param target_weights: list of numpy float vectors to feed as target weights.
        :param bucket_id: which bucket of the model to use.
        :param forward_only: whether to do the backward step or only forward.
        :param return_rnn_hidden_states: if set to True, return the hidden states
            of the two RNNs.
        :return (gradient_norm, average_perplexity, outputs)
        """

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = self.feed_input(formatted_example)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            if bucket_id == -1:
                output_feed = {
                    'updates': self.updates,                    # Update Op that does SGD.
                    'gradient_norms': self.gradient_norms,      # Gradient norm.
                    'losses': self.losses}                      # Loss for this batch.
            else:
                output_feed = {
                    'updates': self.updates[bucket_id],         # Update Op that does SGD.
                    'gradient_norms': self.gradient_norms[bucket_id],  # Gradient norm.
                    'losses': self.losses[bucket_id]}           # Loss for this batch.
        else:
            if bucket_id == -1:
                output_feed = {
                    'output_symbols': self.output_symbols,      # Loss for this batch.
                    'sequence_logits': self.sequence_logits,        # Batch output sequence
                    'losses': self.losses}                      # Batch output scores
            else:
                output_feed = {
                    'output_symbols': self.output_symbols[bucket_id], # Loss for this batch.
                    'sequence_logits': self.sequence_logits[bucket_id],   # Batch output sequence
                    'losses': self.losses[bucket_id]}           # Batch output logits

        if self.tg_token_use_attention:
            if bucket_id == -1:
                output_feed['attn_alignments'] = self.attn_alignments
            else:
                output_feed['attn_alignments'] = self.attn_alignments[bucket_id]

        if bucket_id != -1:
            assert(isinstance(self.encoder_hidden_states, list))
            assert(isinstance(self.decoder_hidden_states, list))
            output_feed['encoder_hidden_states'] = \
                self.encoder_hidden_states[bucket_id]
            output_feed['decoder_hidden_states'] = \
                self.decoder_hidden_states[bucket_id]
        else:
            output_feed['encoder_hidden_states'] = self.encoder_hidden_states
            output_feed['decoder_hidden_states'] = self.decoder_hidden_states

        if self.use_copy:
            output_feed['pointers'] = self.pointers

        extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        if extra_update_ops and not forward_only:
            outputs, extra_updates = session.run(
                [output_feed, extra_update_ops], input_feed)
        else:
            outputs = session.run(output_feed, input_feed)

        O = Output()
        if not forward_only:
            # Gradient norm, loss, no outputs
            O.gradient_norms = outputs['gradient_norms']
            O.losses = outputs['losses']
        else:
            # No gradient loss, output_symbols, sequence_logits
            O.output_symbols = outputs['output_symbols']
            O.sequence_logits = outputs['sequence_logits']
            O.losses = outputs['losses']
        # [attention_masks]
        if self.tg_token_use_attention:
            O.attn_alignments = outputs['attn_alignments']

        O.encoder_hidden_states = outputs['encoder_hidden_states']
        O.decoder_hidden_states = outputs['decoder_hidden_states']

        if self.use_copy:
            O.pointers = outputs['pointers']

        return O













def RNNModel(cell, inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)->Tuple[list,list]:
  """Creates a recurrent neural network specified by RNNCell `cell`.

  The simplest form of RNN network generated is:
    state = cell.zero_state(...)
    outputs = []
    for input_ in inputs:
      output, state = cell(input_, state)
      outputs.append(output)
    return (outputs, state)

  However, a few other options are available:

  An initial state can be provided.
  If the sequence_length vector is provided, dynamic calculation is performed.
  This method of calculation does not compute the RNN steps past the maximum
  sequence length of the minibatch (thus saving computational time),
  and properly propagates the state at an example's sequence length
  to the final state output.

  The dynamic calculation performed is, at time t for batch row b,
    (output, state)(b, t) =
      (t >= sequence_length(b))
        ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
        : cell(input(b, t), state(b, t - 1))

  Args:
    cell: An instance of RNNCell.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, input_size].
    initial_state: (optional) An initial state for the RNN.
      If `cell.state_size` is an integer, this must be
      a tensor of appropriate type and shape `[batch_size x cell.state_size]`.
      If `cell.state_size` is a tuple, this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell.state_size`.
    dtype: (optional) The data type for the initial state.  Required if
      initial_state is not provided.
    sequence_length: Specifies the length of each sequence in inputs.
      An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
    num_cell_layers: Num of layers of the RNN cell. (Mainly used for generating
      output state representations for multi-layer RNN cells.)
    scope: VariableScope for the created subgraph; defaults to "RNN".

  Returns:
    A pair (outputs, state) where:
      - outputs is a length T list of outputs (one for each step)
      - states is a length T list of hidden states (one for each step)

  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If `inputs` is `None` or an empty list, or if the input depth
      (column size) cannot be inferred from inputs via shape inference.
  """

  if not isinstance(cell, tf.compat.v1.nn.rnn_cell.RNNCell):
    raise TypeError("cell must be an instance of RNNCell")
  if not isinstance(inputs, list):
    raise TypeError("inputs must be a list")
  if not inputs:
    raise ValueError("inputs must not be empty")

  outputs = []
  states = []

  # Create a new scope in which the caching device is either
  # determined by the parent scope, or is set to place the cached
  # Variable using the same placement as for the rest of the RNN.
  with tf.compat.v1.variable_scope(scope or "RNN") as varscope:
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)

    # Temporarily avoid EmbeddingWrapper and seq2seq badness
    # TODO(lukaszkaiser): remove EmbeddingWrapper
    if inputs[0].get_shape().ndims != 1:
      (fixed_batch_size, input_size) = inputs[0].get_shape().with_rank(2)
      if input_size is None:
        raise ValueError(
            "Input size (second dimension of inputs[0]) must be accessible via "
            "shape inference, but saw value None.")
    else:
      fixed_batch_size = inputs[0].get_shape().with_rank_at_least(1)[0]

    if fixed_batch_size:
      batch_size = fixed_batch_size
    else:
      batch_size = tf.shape(input=inputs[0])[0]
    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, "
                           "dtype must be specified")
      state = cell.zero_state(batch_size, dtype)

    if sequence_length is not None:  # Prepare variables
      sequence_length = tf.cast(sequence_length, dtype=tf.int32)
      zero_output = tf.zeros(
          tf.stack([batch_size, cell.output_size]), inputs[0].dtype)
      zero_output.set_shape(
          tf.tensor_shape.TensorShape([fixed_batch_size.value,
                                       cell.output_size]))
      min_sequence_length = tf.reduce_min(input_tensor=sequence_length)
      max_sequence_length = tf.reduce_max(input_tensor=sequence_length)

    for time, input_ in enumerate(inputs):
      if time > 0: varscope.reuse_variables()
      # pylint: disable=cell-var-from-loop
      call_cell = lambda: cell(input_, state)
      # pylint: enable=cell-var-from-loop
      if sequence_length is not None:
        (output, state) = tf.nn.rnn._rnn_step(
            time=time, sequence_length=sequence_length,
            min_sequence_length=min_sequence_length,
            max_sequence_length=max_sequence_length,
            zero_output=zero_output, state=state,
            call_cell=call_cell, state_size=cell.state_size)
      else:
        (output, state) = call_cell()

      outputs.append(output)
      states.append(state)
    return (outputs, states)






class Decoder(NNModel):
    def __init__(self, hyperparameters, scope, dim, embedding_dim,
                 use_attention, attention_function, input_keep, output_keep,
                 decoding_algorithm)->None:
        """
        :param hyperparameters: Tellina model hyperparameters.
        :param scope: Scope of the decoder. (There might be multiple decoders
            with the same construction in the neural architecture.)
        :param vocab_size: Output vocabulary size.
        :param dim: Decoder dimension.
        :param embedding_dim: Decoder embedding dimension.
        :param use_attention: Set to True to use attention for decoding.
        :param attention_function: The attention function to use.
        :param input_keep: Dropout parameter for the input of the attention layer.
        :param output_keep: Dropout parameter for the output of the attention layer.
        :param decoding_algorithm: The decoding algorithm to use.
            1. "greedy"
            2. "beam_search"
        """
        super(Decoder, self).__init__(hyperparameters)
        if self.forward_only:
            self.hyperparams['batch_size'] = 1

        self.scope = scope
        self.dim = dim
        self.embedding_dim = embedding_dim
        self.use_attention = use_attention
        self.attention_function = attention_function
        self.input_keep = input_keep
        self.output_keep = output_keep
        self.decoding_algorithm = decoding_algorithm

        self.vocab_size = self.target_vocab_size
        
        # variable sharing
        self.embedding_vars = False
        self.output_project_vars = False

        self.beam_decoder = beam_search.BeamDecoder(
                self.num_layers,
                data_utils.ROOT_ID,
                data_utils.EOS_ID,
                self.batch_size,
                self.beam_size,
                self.use_attention,
                self.use_copy,
                self.copy_fun,
                self.alpha,
                locally_normalized=(self.training_algorithm != "bso")
            ) if self.decoding_algorithm == "beam_search" else None

        self.output_project = self.output_project()

    def embeddings(self):
        with tf.variable_scope(self.scope + "_embeddings", reuse=self.embedding_vars):
            vocab_size = self.target_vocab_size
            print("target token embedding size = {}".format(vocab_size))
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)
            embeddings = tf.get_variable("embedding",
                [vocab_size, self.embedding_dim], initializer=initializer)
            self.embedding_vars = True
            return embeddings

    def token_features(self):
        return np.load(self.tg_token_features_path)

    def output_project(self):
        with tf.variable_scope(self.scope + "_output_project",
                               reuse=self.output_project_vars):
            w = tf.get_variable("proj_w", [self.dim, self.vocab_size])
            b = tf.get_variable("proj_b", [self.vocab_size])
            self.output_project_vars = True
        return (w, b)



class RNNDecoder(Decoder):
    def __init__(self, hyperparameters, scope, dim, embedding_dim, use_attention,
                 attention_function, input_keep, output_keep,
                 decoding_algorithm)->None:
        """
        :member hyperparameters:
        :member scope:
        :member dim:
        :member embedding_dim:
        :member use_attention:
        :member attention_function:
        :member input_keep:
        :member output_keep:
        :member decoding_algorithm:
        """
        super(RNNDecoder, self).__init__(hyperparameters, scope, dim,
            embedding_dim, use_attention, attention_function, input_keep,
            output_keep, decoding_algorithm)
        print("{} dimension = {}".format(scope, dim))
        print("{} decoding_algorithm = {}".format(scope, decoding_algorithm))


    def define_graph(self, encoder_state, decoder_inputs,
                     input_embeddings=None, encoder_attn_masks=None,
                     attention_states=None, num_heads=1,
                     encoder_copy_inputs=None)->Any:
        """
        :param encoder_state: Encoder state => initial decoder state.
        :param decoder_inputs: Decoder training inputs ("<START>, ... <EOS>").
        :param input_embeddings: Decoder vocabulary embedding.
        :param encoder_attn_masks: Binary masks whose entries corresponding to non-padding tokens are 1.
        :param attention_states: Encoder hidden states.
        :param num_heads: Number of attention heads.
        :param encoder_copy_inputs: Array of encoder copy inputs where the copied words are represented using target
            vocab indices and place holding indices are used elsewhere.
        :return output_symbols: (batched) discrete output sequences
        :return output_logits: (batched) output sequence scores
        :return outputs: (batched) output states for all steps
        :return states: (batched) hidden states for all steps
        :return attn_alignments: (batched) attention masks (if attention is used)
        """
        if self.use_attention:
            assert(attention_states.get_shape()[1:2].is_fully_defined())
        if encoder_copy_inputs:
            assert(attention_states.get_shape()[1] == len(encoder_copy_inputs))
        bs_decoding = self.forward_only and \
                      self.decoding_algorithm == "beam_search"

        if input_embeddings is None:
            input_embeddings = self.embeddings()

        if self.force_reading_input:
            print("Warning: reading ground truth decoder inputs at decoding time.")

        with tf.variable_scope(self.scope + "_decoder_rnn") as scope:
            decoder_cell = self.decoder_cell()
            states = []
            alignments_list = []
            pointers = None

            # Cell Wrappers -- 'Attention', 'CopyNet', 'BeamSearch'
            if bs_decoding:
                beam_decoder = self.beam_decoder
                state = beam_decoder.wrap_state(encoder_state, self.output_project)
            else:
                state = encoder_state
                past_output_symbols = []
                past_output_logits = []

            if self.use_attention:
                if bs_decoding:
                    encoder_attn_masks = graph_utils.wrap_inputs(
                        beam_decoder, encoder_attn_masks)
                    attention_states = beam_decoder.wrap_input(attention_states)
                encoder_attn_masks = [tf.expand_dims(encoder_attn_mask, 1)
                    for encoder_attn_mask in encoder_attn_masks]
                encoder_attn_masks = tf.concat(axis=1, values=encoder_attn_masks)
                decoder_cell = decoder.AttentionCellWrapper(
                    decoder_cell,
                    attention_states,
                    encoder_attn_masks,
                    self.attention_function,
                    self.attention_input_keep,
                    self.attention_output_keep,
                    num_heads,
                    self.dim,
                    self.num_layers,
                    self.use_copy,
                    self.vocab_size
                )

            if self.use_copy and self.copy_fun == 'copynet':
                decoder_cell = decoder.CopyCellWrapper(
                    decoder_cell,
                    self.output_project,
                    self.num_layers,
                    encoder_copy_inputs,
                    self.vocab_size)

            if bs_decoding:
                decoder_cell = beam_decoder.wrap_cell(
                    decoder_cell, self.output_project)

            def step_output_symbol_and_logit(output):
                epsilon = tf.constant(1e-12)
                if self.copynet:
                    output_logits = tf.log(output + epsilon)
                else:
                    W, b = self.output_project
                    output_logits = tf.log(
                        tf.nn.softmax(tf.matmul(output, W) + b) + epsilon)
                output_symbol = tf.argmax(output_logits, 1)
                past_output_symbols.append(output_symbol)
                past_output_logits.append(output_logits)
                return output_symbol, output_logits

            for i, input in enumerate(decoder_inputs):
                if bs_decoding:
                    input = beam_decoder.wrap_input(input)

                if i > 0:
                    scope.reuse_variables()
                    if self.forward_only:
                        if self.decoding_algorithm == "beam_search":
                            (
                                past_beam_symbols,  # [batch_size*self.beam_size, max_len], right-aligned!!!
                                past_beam_logprobs, # [batch_size*self.beam_size]
                                past_cell_states,   # [batch_size*self.beam_size, max_len, dim]
                            ) = state
                            input = past_beam_symbols[:, -1]
                        elif self.decoding_algorithm == "greedy":
                            output_symbol, _ = step_output_symbol_and_logit(output)
                            if not self.force_reading_input:
                                input = tf.cast(output_symbol, dtype=tf.int32)
                    else:
                        step_output_symbol_and_logit(output)
                    if self.copynet:
                        decoder_input = input
                        input = tf.where(input >= self.target_vocab_size,
                                         tf.ones_like(input)*data_utils.UNK_ID, input)

                input_embedding = tf.nn.embedding_lookup(input_embeddings, input)

                # Appending selective read information for CopyNet
                if self.copynet:
                    attn_length = attention_states.get_shape()[1]
                    attn_dim = attention_states.get_shape()[2]
                    if i == 0:
                        # Append dummy zero vector to the <START> token
                        selective_reads = tf.zeros([self.batch_size, attn_dim])
                        if bs_decoding:
                            selective_reads = beam_decoder.wrap_input(selective_reads)
                    else:
                        encoder_copy_inputs_2d = tf.concat(
                            [tf.expand_dims(x, 1) for x in encoder_copy_inputs], axis=1)
                        if self.forward_only:
                            copy_input = tf.where(decoder_input >= self.target_vocab_size,
                                                  tf.reduce_sum(
                                                    tf.one_hot(input-self.target_vocab_size, 
                                                               depth=attn_length, dtype=tf.int32)
                                                    * encoder_copy_inputs_2d,
                                                    axis=1),
                                                  decoder_input)
                        else:
                            copy_input = decoder_input
                        tiled_copy_input = tf.tile(input=tf.reshape(copy_input, [-1, 1]),
                                                   multiples=np.array([1, attn_length]))
                        # [batch_size(*self.beam_size), max_source_length]
                        selective_mask = tf.cast(tf.equal(tiled_copy_input, encoder_copy_inputs_2d),
                                                 dtype=tf.float32)
                        # [batch_size(*self.beam_size), max_source_length]
                        weighted_selective_mask = tf.nn.softmax(selective_mask * alignments[1])
                        # [batch_size(*self.beam_size), max_source_length, attn_dim]
                        weighted_selective_mask_3d = tf.tile(input=tf.expand_dims(weighted_selective_mask, 2),
                                                             multiples=np.array([1, 1, attn_dim]))
                        # [batch_size(*self.beam_size), attn_dim]
                        selective_reads = tf.reduce_sum(weighted_selective_mask_3d * attention_states, axis=1)
                    input_embedding = tf.concat([input_embedding, selective_reads], axis=1)

                if self.copynet:
                    output, state, alignments, attns = \
                        decoder_cell(input_embedding, state)
                    alignments_list.append(alignments)
                elif self.use_attention:
                    output, state, alignments, attns = \
                        decoder_cell(input_embedding, state)
                    alignments_list.append(alignments)
                else:
                    output, state = decoder_cell(input_embedding, state)

                # save output states
                if not bs_decoding:
                    # when doing beam search decoding, the output state of each
                    # step cannot simply be gathered step-wise outside the decoder
                    # (speical case: beam_size = 1)
                    states.append(state)

            if self.use_attention:
                # Tensor list --> tenosr
                attn_alignments = tf.concat(axis=1,
                    values=[tf.expand_dims(x[0], 1) for x in alignments_list])
            if self.copynet:
                pointers = tf.concat(axis=1,
                    values=[tf.expand_dims(x[1], 1) for x in alignments_list])

            if bs_decoding:
                # Beam-search output
                (
                    past_beam_symbols,  # [batch_size*self.beam_size, max_len], right-aligned!!!
                    past_beam_logprobs, # [batch_size*self.beam_size]
                    past_cell_states,
                ) = state
                # [self.batch_size, self.beam_size, max_len]
                top_k_osbs = tf.reshape(past_beam_symbols[:, 1:],
                                        [self.batch_size, self.beam_size, -1])
                top_k_osbs = tf.split(axis=0, num_or_size_splits=self.batch_size,
                                      value=top_k_osbs)
                top_k_osbs = [tf.split(axis=0, num_or_size_splits=self.beam_size,
                                       value=tf.squeeze(top_k_output, axis=[0]))
                              for top_k_output in top_k_osbs]
                top_k_osbs = [[tf.squeeze(output, axis=[0]) for output in top_k_output]
                              for top_k_output in top_k_osbs]
                # [self.batch_size, self.beam_size]
                top_k_seq_logits = tf.reshape(past_beam_logprobs,
                                              [self.batch_size, self.beam_size])
                top_k_seq_logits = tf.split(axis=0, num_or_size_splits=self.batch_size,
                                            value=top_k_seq_logits)
                top_k_seq_logits = [tf.squeeze(top_k_logit, axis=[0])
                                    for top_k_logit in top_k_seq_logits]
                if self.use_attention:
                    attn_alignments = tf.reshape(attn_alignments,
                            [self.batch_size, self.beam_size, len(decoder_inputs),
                             attention_states.get_shape()[1].value])
                # LSTM: ([batch_size*self.beam_size, :, dim],
                #        [batch_size*self.beam_size, :, dim])
                # GRU: [batch_size*self.beam_size, :, dim]
                if self.rnn_cell == 'lstm':
                    if self.num_layers == 1:
                        c_states, h_states = past_cell_states
                        states = list(zip(
                            [tf.squeeze(x, axis=[1])
                                for x in tf.split(c_states, c_states.get_shape()[1],
                                                  axis=1)],
                            [tf.squeeze(x, axis=[1])
                                for x in tf.split(h_states, h_states.get_shape()[1],
                                                  axis=1)]))
                    else:
                        layered_states = [list(zip(
                                [tf.squeeze(x, axis=[1]) 
                                    for x in tf.split(c_states, c_states.get_shape()[1],
                                                      axis=1)[1:]],
                                [tf.squeeze(x, axis=[1])
                                    for x in tf.split(h_states, h_states.get_shape()[1],
                                                      axis=1)[1:]]))
                            for c_states, h_states in past_cell_states]
                        states = list(zip(layered_states))
                elif self.rnn_cell in ['gru', 'ran']:
                    states = [tf.squeeze(x, axis=[1]) for x in \
                        tf.split(num_or_size_splits=past_cell_states.get_shape()[1],
                                 axis=1, value=past_cell_states)][1:]
                else:
                    raise AttributeError(
                        "Unrecognized rnn cell type: {}".format(self.rnn_cell))
                return top_k_osbs, top_k_seq_logits, states, \
                       states, attn_alignments, pointers
            else:
                # Greedy output
                step_output_symbol_and_logit(output)
                output_symbols = tf.concat(
                    [tf.expand_dims(x, 1) for x in past_output_symbols], axis=1)
                sequence_logits = tf.add_n([tf.reduce_max(x, axis=1) 
                                            for x in past_output_logits])
                return output_symbols, sequence_logits, past_output_logits, \
                       states, attn_alignments, pointers


    def decoder_cell(self):
        if self.copynet:
            input_size = self.dim * 2
        else:
            input_size = self.dim
        with tf.variable_scope(self.scope + "_decoder_cell") as scope:
            cell = create_multilayer_cell(
                self.rnn_cell, scope, self.dim, self.num_layers,
                self.input_keep, self.output_keep,
                variational_recurrent=self.variational_recurrent_dropout,
                batch_normalization=self.recurrent_batch_normalization,
                forward_only=self.forward_only,
                input_dim=input_size)
        return cell
























class Encoder(NNModel):
    def __init__(self, hyperparameters, input_keep, output_keep):
        super(Encoder, self).__init__(hyperparameters)

        # variable reuse
        self.char_embedding_vars = False
        self.token_embedding_vars = False
        self.char_rnn_vars = False

        self.input_keep = input_keep
        self.output_keep = output_keep

        self.channels = []
        self.dim = 0
        if self.sc_token:
            self.channels.append('token')
            self.dim += self.sc_token_dim
        if self.sc_char:
            assert False, 'broken'
            self.channels.append('char')
            self.dim += self.sc_char_dim

        assert(len(self.channels) > 0)

    def token_representations(self, channel_inputs):
        """
        Generate token representations based on multi-channel input.

        :param channel_inputs: an array of channel input indices
            1. batch token indices
            2. batch char indices
        """
        channel_embeddings = []
        if self.sc_token:
            token_embeddings = self.token_embeddings()
            token_channel_embeddings = \
                [tf.nn.embedding_lookup(params=token_embeddings, ids=encoder_input)
                 for encoder_input in channel_inputs[0]]
            channel_embeddings.append(token_channel_embeddings)

        if self.sc_char:
            assert False, 'char channel is broken'
            char_channel_embeddings = \
                self.char_channel_embeddings(channel_inputs[1])
            channel_embeddings.append(char_channel_embeddings)

        if len(channel_embeddings) == 1:
            input_embeddings = channel_embeddings[0]
        else:
            input_embeddings = \
                [tf.concat(axis=1, values=[x, y]) for (x, y) in
                    map(lambda x,y:(x,y), channel_embeddings[0],
                        channel_embeddings[1])]

        return input_embeddings

    def token_embeddings(self):
        """
        Generate token representations by plain table look-up

        :return: token embedding matrix [source_vocab_size, dim]
        """
        with tf.compat.v1.variable_scope("encoder_token_embeddings",
                               reuse=self.token_embedding_vars):
            vocab_size = self.source_vocab_size
            print("source token embedding size = {}".format(vocab_size))
            sqrt3 = math.sqrt(3)
            initializer = tf.compat.v1.random_uniform_initializer(-sqrt3, sqrt3)
            embeddings = tf.compat.v1.get_variable("embedding",
                [vocab_size, self.sc_token_dim], initializer=initializer)
            self.token_embedding_vars = True
            return embeddings


    def char_embeddings(self):
        with tf.compat.v1.variable_scope("encoder_char_embeddings",
                               reuse=self.char_embedding_vars):
            sqrt3 = math.sqrt(3)
            initializer = tf.compat.v1.random_uniform_initializer(-sqrt3, sqrt3)
            embeddings = tf.compat.v1.get_variable(
                "embedding", [self.source_char_vocab_size, self.sc_char_dim],
                initializer=initializer)
            self.char_embedding_vars = True
            return embeddings


    def token_channel_embeddings(self):
        input = self.token_features()
        return tf.nn.embedding_lookup(params=self.token_embeddings(), ids=input)

    def char_channel_embeddings(self, channel_inputs):
        """
        Generate token representations by character composition.

        :param channel_inputs: batch input char indices
                [[batch, token_size], [batch, token_size], ...]
        :return: embeddings_char [source_vocab_size, char_channel_dim]
        """
        inputs = [tf.squeeze(x, 1) for x in tf.split(axis=1,
                  num_or_size_splits=self.max_source_token_size,
                  value=tf.concat(axis=0, values=channel_inputs))]
        input_embeddings = [tf.nn.embedding_lookup(params=self.char_embeddings(), ids=input) 
                            for input in inputs]

        if self.sc_char_composition == 'rnn':
            with tf.compat.v1.variable_scope("encoder_char_rnn",
                                   reuse=self.char_rnn_vars) as scope:
                cell = create_multilayer_cell(
                    self.sc_char_rnn_cell, scope,
                    self.sc_char_dim, self.sc_char_rnn_num_layers,
                    variational_recurrent=self.variational_recurrent_dropout)
                rnn_outputs, rnn_states = RNNModel(cell, input_embeddings,
                                                               dtype=tf.float32)
                self.char_rnn_vars = True
        else:
            raise NotImplementedError

        return [tf.squeeze(x, 0) for x in
                tf.split(axis=0, num_or_size_splits=len(channel_inputs),
                    value=tf.reshape(rnn_states[-1],
                        [len(channel_inputs), -1, self.sc_char_dim]))]

    def token_features(self):
        return np.load(self.sc_token_features_path)

    def token_char_index_matrix(self):
        return np.load(self.sc_char_features_path)




class RNNEncoder(Encoder):
    def __init__(self, hyperparameters, input_keep, output_keep):
        super(RNNEncoder, self).__init__(
            hyperparameters, input_keep, output_keep)
        self.cell = self.encoder_cell()
        self.output_dim = self.dim

    def define_graph(self, encoder_channel_inputs, input_embeddings=None):
        # Compute the continuous input representations
        if input_embeddings is None:
            input_embeddings = self.token_representations(encoder_channel_inputs)
        with tf.compat.v1.variable_scope("encoder_rnn"):
            return RNNModel(self.cell, input_embeddings, dtype=tf.float32)

    def encoder_cell(self):
        """RNN cell for the encoder."""
        with tf.compat.v1.variable_scope("encoder_cell") as scope:
            cell = create_multilayer_cell(self.rnn_cell, scope,
                self.dim, self.num_layers, self.input_keep, self.output_keep,
                variational_recurrent=self.variational_recurrent_dropout)
        return cell




class Seq2SeqModel(EncoderDecoderModel):
    """Sequence-to-sequence model with attention and for multiple buckets.

    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
      http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
      http://arxiv.org/abs/1412.2007
    """

    def __init__(self, hyperparams, buckets=None):
        super(Seq2SeqModel, self).__init__(hyperparams, buckets)


    def define_encoder(self, input_keep, output_keep):
        """
        Construct sequence encoder.
        """
        assert self.encoder_topology == "rnn"
        self.encoder = encoder.RNNEncoder(
            self.hyperparams, input_keep, output_keep)


    def define_decoder(self, dim, embedding_dim, use_attention,
            attention_function, input_keep, output_keep):
        """
        Construct sequence decoder.
        """
        assert self.decoder_topology == "rnn"

        self.decoder = RNNDecoder(
            hyperparameters=self.hyperparams,
            scope='token_decoder', dim=dim,
            embedding_dim=embedding_dim,
            use_attention=use_attention,
            attention_function=attention_function,
            input_keep=input_keep,
            output_keep=output_keep,
            decoding_algorithm=self.token_decoding_algorithm
        )






def define_model(FLAGS, session, model_constructor, buckets, forward_only):
    params = collections.defaultdict()

    params["source_vocab_size"] = FLAGS.sc_vocab_size
    params["target_vocab_size"] = FLAGS.tg_vocab_size
    params["max_source_length"] = FLAGS.max_sc_length
    params["max_target_length"] = FLAGS.max_tg_length
    params["max_source_token_size"] = FLAGS.max_sc_token_size
    params["max_target_token_size"] = FLAGS.max_tg_token_size
    params["rnn_cell"] = FLAGS.rnn_cell
    params["batch_size"] = FLAGS.batch_size
    params["num_layers"] = FLAGS.num_layers
    params["num_samples"] = FLAGS.num_samples
    params["max_gradient_norm"] = FLAGS.max_gradient_norm
    params["variational_recurrent_dropout"] = \
        FLAGS.variational_recurrent_dropout

    params["recurrent_batch_normalization"] = \
        FLAGS.recurrent_batch_normalization
    params["gramma_c"] = FLAGS.gamma_c
    params["beta_c"] = FLAGS.beta_c
    params["gramma_h"] = FLAGS.gamma_h
    params["beta_h"] = FLAGS.beta_h
    params["gramma_x"] = FLAGS.gamma_x
    params["beta_x"] = FLAGS.beta_x

    params["tg_token_use_attention"] = FLAGS.tg_token_use_attention

    params["sc_token"] = FLAGS.sc_token
    params["sc_token_dim"] = FLAGS.sc_token_dim
    params["sc_char"] = FLAGS.sc_char
    # params["sc_char_vocab_size"] = FLAGS.sc_char_vocab_size
    # params["sc_char_dim"] = FLAGS.sc_char_dim
    # params["sc_char_composition"] = FLAGS.sc_char_composition
    # params["sc_char_rnn_cell"] = FLAGS.sc_char_rnn_cell
    # params["sc_char_rnn_num_layers"] = FLAGS.sc_char_rnn_num_layers
    # params["sc_token_features_path"] = os.path.join(
    #     FLAGS.data_dir, "{}.vocab.token.feature.npy".format(source))
    # params["sc_char_features_path"] = os.path.join(
    #     FLAGS.data_dir, "{}.vocab.char.feature.npy".format(source))

    params["tg_token"] = FLAGS.tg_token
    params["tg_char"] = FLAGS.tg_char
    # params["tg_char_vocab_size"] = FLAGS.tg_char_vocab_size
    # params["tg_char_composition"] = FLAGS.tg_char_composition
    # params["tg_char_use_attention"] = FLAGS.tg_char_use_attention
    # params["tg_char_rnn_cell"] = FLAGS.tg_char_rnn_cell
    # params["tg_char_rnn_num_layers"] = FLAGS.tg_char_rnn_num_layers
    # params["tg_char_rnn_input_keep"] = FLAGS.tg_char_rnn_input_keep
    # params["tg_char_rnn_output_keep"] = FLAGS.tg_char_rnn_output_keep
    # params["tg_token_features_path"] = os.path.join(
    #     FLAGS.data_dir, "{}.vocab.token.feature.npy".format(target))
    # params["tg_char_features_path"] = os.path.join(
    #     FLAGS.data_dir, "{}.vocab.char.feature.npy".format(target))

    params["gamma"] = FLAGS.gamma

    params["optimizer"] = FLAGS.optimizer
    params["learning_rate"] = FLAGS.learning_rate
    params["learning_rate_decay_factor"] = FLAGS.learning_rate_decay_factor
    params["adam_epsilon"] = FLAGS.adam_epsilon

    params["steps_per_epoch"] = FLAGS.steps_per_epoch
    params["num_epochs"] = FLAGS.num_epochs

    params["training_algorithm"] = FLAGS.training_algorithm
    if FLAGS.training_algorithm == "bso":
        assert(FLAGS.token_decoding_algorithm == "beam_search")
    params["margin"] = FLAGS.margin

    params["use_copy"] = FLAGS.use_copy
    params["copy_fun"] = FLAGS.copy_fun
    params["chi"] = FLAGS.chi

    params["tg_token_attn_fun"] = FLAGS.tg_token_attn_fun
    params["beta"] = FLAGS.beta

    params["encoder_topology"] = FLAGS.encoder_topology
    params["decoder_topology"] = FLAGS.decoder_topology

    params["sc_input_keep"] = FLAGS.sc_input_keep
    params["sc_output_keep"] = FLAGS.sc_output_keep
    params["tg_input_keep"] = FLAGS.tg_input_keep
    params["tg_output_keep"] = FLAGS.tg_output_keep
    params["attention_input_keep"] = FLAGS.attention_input_keep
    params["attention_output_keep"] = FLAGS.attention_output_keep

    params["token_decoding_algorithm"] = FLAGS.token_decoding_algorithm
    params["char_decoding_algorithm"] = FLAGS.char_decoding_algorithm
    params["beam_size"] = FLAGS.beam_size
    params["alpha"] = FLAGS.alpha
    params["top_k"] = FLAGS.top_k

    params["forward_only"] = forward_only
    params["force_reading_input"] = FLAGS.force_reading_input

    # construct model directory
    model_subdir, decode_sig = get_decode_signature(FLAGS)
    FLAGS.model_dir = os.path.join(FLAGS.model_root_dir, model_subdir)
    params["model_dir"] = FLAGS.model_dir
    params["decode_sig"] = decode_sig
    print("model_dir={}".format(FLAGS.model_dir))
    print("decode_sig={}".format(decode_sig))

    if FLAGS.gen_slot_filling_training_data:
        FLAGS.batch_size = 1
        params["batch_size"] = 1
        FLAGS.beam_size = 1
        params["beam_size"] = 1
        FLAGS.learning_rate = 0
        params["learning_rate"] = 0
        params["force_reading_input"] = True
        params["create_fresh_params"] = False

    if FLAGS.explain:
        FLAGS.grammatical_only = False

    model = model_constructor(params, buckets)

    # if forward_only or FLAGS.gen_slot_filling_training_data or \
    #         not FLAGS.create_fresh_params:
    #     ckpt = tf.train.get_checkpoint_state(
    #         os.path.join(FLAGS.model_root_dir, FLAGS.model_dir))
    #     print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    #     model.saver.restore(session, ckpt.model_checkpoint_path)
    # else:
    if not os.path.exists(FLAGS.model_dir):
      print("Making model_dir...")
      os.mkdir(FLAGS.model_dir)

    print("Initialize the graph with random parameters.")
    session.run(tf.compat.v1.global_variables_initializer())

    return model
