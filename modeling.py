# from typing import Union, List, Callable
#
# import tensorflow as tf
#
# import layers
# import rnn_cell
#
# CELL_TYPES = ['LRAN', 'RAN', 'LSTM', 'GRU']
#
#
# class CANTRIPModel(object):
#     def __init__(self,
#                  max_seq_len: int,
#                  max_snapshot_size: int,
#                  vocabulary_size: int,
#                  observation_embedding_size: int,
#                  delta_encoding_size: int,
#                  num_hidden: Union[int, List[int]],
#                  cell_type: str,
#                  batch_size: int,
#                  snapshot_encoder: Callable[['CANTRIPModel'], tf.Tensor],
#                  dropout: float = 0.,
#                  vocab_dropout: float = None,
#                  num_classes: int = 2,
#                  delta_combine: str = "concat",
#                  embed_delta: bool = False,
#                  rnn_highway_depth: int = 3,
#                  rnn_direction='forward'):
#         self.max_seq_len = max_seq_len
#         self.max_snapshot_size = max_snapshot_size
#         self.vocabulary_size = vocabulary_size
#         self.embedding_size = observation_embedding_size
#         self.num_hidden = num_hidden
#         self.batch_size = batch_size
#         self.num_classes = num_classes
#         self.delta_encoding_size = delta_encoding_size
#         self.dropout = dropout
#         self.vocab_dropout = vocab_dropout or dropout
#         self.cell_type = cell_type
#         self.delta_combine = delta_combine
#         self.embed_delta = embed_delta
#         self.rnn_highway_depth = rnn_highway_depth
#         self.rnn_direction = rnn_direction
#
#         if delta_combine == 'add' and not self.embed_delta and self.embedding_size != self.delta_encoding_size:
#             print("Cannot add delta embeddings of size %d to observation encodings of size %d, "
#                   "setting embed_delta=True" %
#                   (self.delta_encoding_size, self.embedding_size))
#             self.embed_delta = True
#         with tf.name_scope('cantrip'):
#             self._add_placeholders()
#             with tf.name_scope('snapshot_encoder'):  # , regularizer=self.regularizer):
#                 self.snapshot_encodings = snapshot_encoder(self)
#
#             if self.embed_delta:
#                 with tf.name_scope('delta_encoder'):
#                     self.delta_inputs = tf.keras.layers.Dense(units=self.embedding_size,
#                                                               activation=None,
#                                                               name='delta_embeddings')(self.deltas)
#             else:
#                 self.delta_inputs = self.deltas
#
#             self._add_seq_rnn(cell_type)
#             # Convert to sexy logits
#             self.logits = tf.keras.layers.Dense(units=self.num_classes,
#                                                 activation=None,
#                                                 name='class_logits')(self.seq_final_output)
#         self._add_postprocessing()
#
#     # def _add_placeholders(self):
#     #     self.observations = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len, self.max_snapshot_size],
#     #                                        name="observations")
#     #
#     #     # Elapsed time deltas
#     #     self.deltas = tf.placeholder(tf.float32, [self.batch_size, self.max_seq_len, self.delta_encoding_size],
#     #                                  name="deltas")
#     #
#     #     # Snapshot sizes
#     #     self.snapshot_sizes = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len], name="snapshot_sizes")
#     #
#     #     # Chronology lengths
#     #     self.seq_lengths = tf.placeholder(tf.int32, [self.batch_size], name="seq_lengths")
#     #
#     #     # Label
#     #     self.labels = tf.placeholder(tf.int32, [self.batch_size], name="labels")
#     #
#     #     # Training
#     #     self.training = tf.placeholder(tf.bool, name="training")
#     def _add_placeholders(self):
#         # Define input layers using tf.keras.Input
#         self.observations = tf.keras.Input(shape=(self.max_seq_len, self.max_snapshot_size), dtype=tf.int32,
#                                            name="observations")
#
#         # Elapsed time deltas
#         self.deltas = tf.keras.Input(shape=(self.max_seq_len, self.delta_encoding_size), dtype=tf.float32,
#                                      name="deltas")
#
#         # Snapshot sizes
#         self.snapshot_sizes = tf.keras.Input(shape=(self.max_seq_len,), dtype=tf.int32, name="snapshot_sizes")
#
#         # Chronology lengths
#         self.seq_lengths = tf.keras.Input(shape=(self.batch_size,), dtype=tf.int32, name="seq_lengths")
#
#         # Label
#         self.labels = tf.keras.Input(shape=(self.batch_size,), dtype=tf.int32, name="labels")
#
#         # Training
#         self.training = tf.keras.Input(shape=(), dtype=tf.bool, name="training")
#
#     def _add_seq_rnn(self, cell_type: str):
#         with tf.name_scope('sequence'):
#             # Add dropout on deltas
#             if self.dropout > 0:
#                 self.delta_inputs = tf.keras.layers.Dropout(rate=self.dropout)(self.delta_inputs,training=self.training)
#             # Concat observation_t and delta_t (deltas are already shifted by one)
#             if self.delta_combine == 'concat':
#                 self.x = tf.concat([self.snapshot_encodings, self.delta_inputs], axis=-1, name='rnn_input_concat')
#             elif self.delta_combine == 'add':
#                 self.x = self.snapshot_encodings + self.delta_inputs
#             else:
#                 raise ValueError("Invalid delta combination method: %s" % self.delta_combine)
#             if self.dropout > 0:
#                 self.x = tf.keras.layers.Dropout(rate=self.dropout)(self.x, training=self.training)
#             _cell_types = {
#                 # Original RAN from https://arxiv.org/abs/1705.07393
#                 'RAN': rnn_cell.RANCell,
#                 'RAN-LN': lambda units: rnn_cell.RANCell(units, normalize=True),
#                 'VHRAN': lambda units: rnn_cell.VHRANCell(units, self.x.shape[-1], depth=self.rnn_highway_depth),
#                 'VHRAN-LN': lambda units: rnn_cell.VHRANCell(units, self.x.shape[-1], depth=self.rnn_highway_depth,
#                                                              normalize=True),
#                 'RHN': lambda units: rnn_cell.RHNCell(units, self.x.shape[-1],
#                                                       depth=self.rnn_highway_depth,
#                                                       is_training=self.training),
#                 'RHN-LN': lambda units: rnn_cell.RHNCell(units, self.x.shape[-1],
#                                                          depth=self.rnn_highway_depth,
#                                                          is_training=self.training,
#                                                          normalize=True),
#                 'LSTM': tf.nn.rnn_cell.BasicLSTMCell,
#                 'LSTM-LN': tf.contrib.rnn.LayerNormBasicLSTMCell,
#                 'GRU': tf.nn.rnn_cell.GRUCell,
#                 'GRU-LN': rnn_cell.LayerNormGRUCell
#             }
#             if cell_type not in _cell_types:
#                 raise ValueError('unsupported cell type %s', cell_type)
#             self.cell_fn = _cell_types[cell_type]
#             if self.rnn_direction == 'bidirectional':
#                 self.seq_final_output = layers.bidirectional_rnn_layer(self.cell_fn,self.num_hidden, self.x, self.seq_lengths)
#             else:
#                 self.seq_final_output = layers.rnn_layer(self.cell_fn, self.num_hidden, self.x, self.seq_lengths)
#             print('Final output:', self.seq_final_output)
#             if self.dropout > 0:
#                 self.seq_final_output = \
#                     tf.keras.layers.Dropout(rate=self.dropout)(self.seq_final_output, training=self.training)
#     def _add_postprocessing(self):
#         """Categorical arg-max prediction for disease-risk"""
#         # Class labels (used mainly for metrics)
#         self.y = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='class_predictions')
# # from typing import Union, List, Callable
# # import tensorflow as tf
# # import layers  # 确保这是您自定义的模块
# # import rnn_cell  # 确保这是您自定义的模块
# #
# # CELL_TYPES = ['LRAN', 'RAN', 'LSTM', 'GRU']
# #
# # class CANTRIPModel(tf.keras.Model):
# #     def __init__(self,
# #                  max_seq_len: int,
# #                  max_snapshot_size: int,
# #                  vocabulary_size: int,
# #                  observation_embedding_size: int,
# #                  delta_encoding_size: int,
# #                  num_hidden: Union[int, List[int]],
# #                  cell_type: str,
# #                  batch_size: int,
# #                  snapshot_encoder: Callable[['CANTRIPModel'], tf.Tensor],
# #                  dropout: float = 0.,
# #                  vocab_dropout: float = None,
# #                  num_classes: int = 2,
# #                  delta_combine: str = "concat",
# #                  embed_delta: bool = False,
# #                  rnn_highway_depth: int = 3,
# #                  rnn_direction='forward'):
# #         super(CANTRIPModel, self).__init__()
# #         self.max_seq_len = max_seq_len
# #         self.max_snapshot_size = max_snapshot_size
# #         self.vocabulary_size = vocabulary_size
# #         self.embedding_size = observation_embedding_size
# #         self.num_hidden = num_hidden
# #         self.batch_size = batch_size
# #         self.num_classes = num_classes
# #         self.delta_encoding_size = delta_encoding_size
# #         self.dropout = dropout
# #         self.vocab_dropout = vocab_dropout or dropout
# #         self.cell_type = cell_type
# #         self.delta_combine = delta_combine
# #         self.embed_delta = embed_delta
# #         self.rnn_highway_depth = rnn_highway_depth
# #         self.rnn_direction = rnn_direction
# #
# #         if delta_combine == 'add' and not self.embed_delta and self.embedding_size != self.delta_encoding_size:
# #             print("Cannot add delta embeddings of size %d to observation encodings of size %d, "
# #                   "setting embed_delta=True" %
# #                   (self.delta_encoding_size, self.embedding_size))
# #             self.embed_delta = True
# #
# #         # Input layers
# #         self.observations = tf.keras.Input(shape=(self.max_seq_len, self.max_snapshot_size), name="observations")
# #         self.deltas = tf.keras.Input(shape=(self.max_seq_len, self.delta_encoding_size), name="deltas")
# #         self.snapshot_sizes = tf.keras.Input(shape=(self.max_seq_len,), name="snapshot_sizes")
# #         self.seq_lengths = tf.keras.Input(shape=(self.batch_size,), name="seq_lengths")
# #         self.labels = tf.keras.Input(shape=(self.batch_size,), name="labels")
# #         self.training = tf.keras.Input(shape=(), dtype=tf.bool, name="training")
# #
# #         # Build the model
# #         self._build_model(snapshot_encoder)
# #
# #     def _build_model(self, snapshot_encoder):
# #         # Snapshot Encoder
# #         self.snapshot_encodings = snapshot_encoder(self)
# #
# #         # Delta Encoder
# #         if self.embed_delta:
# #             self.delta_inputs = tf.keras.layers.Dense(units=self.embedding_size, activation=None, name='delta_embeddings')(self.deltas)
# #         else:
# #             self.delta_inputs = self.deltas
# #
# #         # RNN Sequence
# #         self._add_seq_rnn(self.cell_type)
# #
# #         # Output layer for logits
# #         self.logits = tf.keras.layers.Dense(units=self.num_classes, activation=None, name='class_logits')(self.seq_final_output)
# #
# #         # Post-processing
# #         self._add_postprocessing()
# #
# #     def _add_seq_rnn(self, cell_type: str):
# #         # Add dropout on deltas
# #         if self.dropout > 0:
# #             self.delta_inputs = tf.keras.layers.Dropout(rate=self.dropout)(self.delta_inputs, training=self.training)
# #
# #         # Combine snapshot encodings and delta inputs
# #         if self.delta_combine == 'concat':
# #             self.x = tf.concat([self.snapshot_encodings, self.delta_inputs], axis=-1, name='rnn_input_concat')
# #         elif self.delta_combine == 'add':
# #             self.x = self.snapshot_encodings + self.delta_inputs
# #         else:
# #             raise ValueError("Invalid delta combination method: %s" % self.delta_combine)
# #
# #         if self.dropout > 0:
# #             self.x = tf.keras.layers.Dropout(rate=self.dropout)(self.x, training=self.training)
# #
# #         # Define RNN cell types
# #         _cell_types = {
# #             'RAN': rnn_cell.RANCell,
# #             'RAN-LN': lambda units: rnn_cell.RANCell(units, normalize=True),
# #             'VHRAN': lambda units: rnn_cell.VHRANCell(units, self.x.shape[-1], depth=self.rnn_highway_depth),
# #             'VHRAN-LN': lambda units: rnn_cell.VHRANCell(units, self.x.shape[-1], depth=self.rnn_highway_depth, normalize=True),
# #             'RHN': lambda units: rnn_cell.RHNCell(units, self.x.shape[-1], depth=self.rnn_highway_depth, is_training=self.training),
# #             'RHN-LN': lambda units: rnn_cell.RHNCell(units, self.x.shape[-1], depth=self.rnn_highway_depth, is_training=self.training, normalize=True),
# #             'LSTM': tf.keras.layers.LSTMCell,
# #             'GRU': tf.keras.layers.GRUCell,
# #         }
# #
# #         if cell_type not in _cell_types:
# #             raise ValueError('unsupported cell type %s' % cell_type)
# #
# #         self.cell_fn = _cell_types[cell_type]
# #
# #         if self.rnn_direction == 'bidirectional':
# #             self.seq_final_output = layers.bidirectional_rnn_layer(self.cell_fn, self.num_hidden, self.x, self.seq_lengths)
# #         else:
# #             self.seq_final_output = layers.rnn_layer(self.cell_fn, self.num_hidden, self.x, self.seq_lengths)
# #
# #         if self.dropout > 0:
# #             self.seq_final_output = tf.keras.layers.Dropout(rate=self.dropout)(self.seq_final_output, training=self.training)
# #
# #     def _add_postprocessing(self):
# #         """Categorical arg-max prediction for disease-risk"""
# #         # Class labels (used mainly for metrics)
# #         self.y = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='class_predictions')
#
# # # 使用示例
# # model = CANTRIPModel(
# #     max_seq_len=10,
# #     max_snapshot_size=20,
# #     vocabulary_size=1000,
# #     observation_embedding_size=128,
# #     delta_encoding_size=1,
# #     num_hidden=64,
# #     cell_type='LSTM',
# #     batch_size=32,
# #     snapshot_encoder=lambda x: tf.keras.layers.Dense(128)(x.observations)
# # )

# from typing import Union, List, Callable
#
# import tensorflow as tf
#
# import layers
# import rnn_cell
#
# CELL_TYPES = ['LRAN', 'RAN', 'LSTM', 'GRU']
#
#
# class CANTRIPModel(object):
#     def __init__(self,
#                  max_seq_len: int,
#                  max_snapshot_size: int,
#                  vocabulary_size: int,
#                  observation_embedding_size: int,
#                  delta_encoding_size: int,
#                  num_hidden: Union[int, List[int]],
#                  cell_type: str,
#                  batch_size: int,
#                  snapshot_encoder: Callable[['CANTRIPModel'], tf.Tensor],
#                  dropout: float = 0.,
#                  vocab_dropout: float = None,
#                  num_classes: int = 2,
#                  delta_combine: str = "concat",
#                  embed_delta: bool = False,
#                  rnn_highway_depth: int = 3,
#                  rnn_direction='forward'):
#         self.max_seq_len = max_seq_len
#         self.max_snapshot_size = max_snapshot_size
#         self.vocabulary_size = vocabulary_size
#         self.embedding_size = observation_embedding_size
#         self.num_hidden = num_hidden
#         self.batch_size = batch_size
#         self.num_classes = num_classes
#         self.delta_encoding_size = delta_encoding_size
#         self.dropout = dropout
#         self.vocab_dropout = vocab_dropout or dropout
#         self.cell_type = cell_type
#         self.delta_combine = delta_combine
#         self.embed_delta = embed_delta
#         self.rnn_highway_depth = rnn_highway_depth
#         self.rnn_direction = rnn_direction
#
#         if delta_combine == 'add' and not self.embed_delta and self.embedding_size != self.delta_encoding_size:
#             print("Cannot add delta embeddings of size %d to observation encodings of size %d, "
#                   "setting embed_delta=True" %
#                   (self.delta_encoding_size, self.embedding_size))
#             self.embed_delta = True
#         with tf.variable_scope('cantrip'):
#             self._add_placeholders()
#             with tf.variable_scope('snapshot_encoder'):  # , regularizer=self.regularizer):
#                 self.snapshot_encodings = snapshot_encoder(self)
#
#             if self.embed_delta:
#                 with tf.variable_scope('delta_encoder'):
#                     self.delta_inputs = tf.keras.layers.Dense(units=self.embedding_size,
#                                                               activation=None,
#                                                               name='delta_embeddings')(self.deltas)
#             else:
#                 self.delta_inputs = self.deltas
#
#             self._add_seq_rnn(cell_type)
#             self.logits = tf.keras.layers.Dense(units=self.num_classes,
#                                                 activation=None,
#                                                 name='class_logits')(self.seq_final_output)
#         self._add_postprocessing()
#     def _add_placeholders(self):
#         # Observation IDs
#         self.observations = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len, self.max_snapshot_size],
#                                            name="observations")
#
#         # Elapsed time deltas
#         self.deltas = tf.placeholder(tf.float32, [self.batch_size, self.max_seq_len, self.delta_encoding_size],
#                                      name="deltas")
#
#         # Snapshot sizes
#         self.snapshot_sizes = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len], name="snapshot_sizes")
#
#         # Chronology lengths
#         self.seq_lengths = tf.placeholder(tf.int32, [self.batch_size], name="seq_lengths")
#
#         # Label
#         self.labels = tf.placeholder(tf.int32, [self.batch_size], name="labels")
#
#         # Training
#         self.training = tf.placeholder(tf.bool, name="training")
#
#     def _add_seq_rnn(self, cell_type: str):
#         """Add the clinical picture inference module; implemented in as an RNN. """
#         with tf.variable_scope('sequence'):
#             # Add dropout on deltas
#             if self.dropout > 0:
#                 self.delta_inputs = tf.keras.layers.Dropout(rate=self.dropout)(self.delta_inputs,
#                                                                                training=self.training)
#
#             # Concat observation_t and delta_t (deltas are already shifted by one)
#             if self.delta_combine == 'concat':
#                 self.x = tf.concat([self.snapshot_encodings, self.delta_inputs], axis=-1, name='rnn_input_concat')
#             elif self.delta_combine == 'add':
#                 self.x = self.snapshot_encodings + self.delta_inputs
#             else:
#                 raise ValueError("Invalid delta combination method: %s" % self.delta_combine)
#
#             # Add dropout on concatenated inputs
#             if self.dropout > 0:
#                 self.x = tf.keras.layers.Dropout(rate=self.dropout)(self.x, training=self.training)
#
#             _cell_types = {
#                 # Original RAN from https://arxiv.org/abs/1705.07393
#                 'RAN': rnn_cell.RANCell,
#                 'RAN-LN': lambda units: rnn_cell.RANCell(units, normalize=True),
#                 'VHRAN': lambda units: rnn_cell.VHRANCell(units, self.x.shape[-1], depth=self.rnn_highway_depth),
#                 'VHRAN-LN': lambda units: rnn_cell.VHRANCell(units, self.x.shape[-1], depth=self.rnn_highway_depth,
#                                                              normalize=True),
#                 'RHN': lambda units: rnn_cell.RHNCell(units, self.x.shape[-1],
#                                                       depth=self.rnn_highway_depth,
#                                                       is_training=self.training),
#                 'RHN-LN': lambda units: rnn_cell.RHNCell(units, self.x.shape[-1],
#                                                          depth=self.rnn_highway_depth,
#                                                          is_training=self.training,
#                                                          normalize=True),
#                 # Super secret simplified RAN variant from Eq. group (2) in https://arxiv.org/abs/1705.07393
#                 # 'LRAN': lambda num_cells: rnn_cell.SimpleRANCell(self.x.shape[-1]),
#                 # 'LRAN-LN': lambda num_cells: rnn_cell.SimpleRANCell(self.x.shape[-1], normalize=True),
#                 'LSTM': tf.nn.rnn_cell.BasicLSTMCell,
#                 'LSTM-LN': tf.contrib.rnn.LayerNormBasicLSTMCell,
#                 'GRU': tf.nn.rnn_cell.GRUCell,
#                 'GRU-LN': rnn_cell.LayerNormGRUCell
#             }
#
#             if cell_type not in _cell_types:
#                 raise ValueError('unsupported cell type %s', cell_type)
#
#             self.cell_fn = _cell_types[cell_type]
#
#             if self.rnn_direction == 'bidirectional':
#                 self.seq_final_output = layers.bidirectional_rnn_layer(self.cell_fn,
#                                                                        self.num_hidden, self.x, self.seq_lengths)
#             else:
#                 self.seq_final_output = layers.rnn_layer(self.cell_fn, self.num_hidden, self.x, self.seq_lengths)
#
#             print('Final output:', self.seq_final_output)
#
#             # Even more fun dropout
#             if self.dropout > 0:
#                 self.seq_final_output = \
#                     tf.keras.layers.Dropout(rate=self.dropout)(self.seq_final_output, training=self.training)
#
#     def _add_postprocessing(self):
#         """Categorical arg-max prediction for disease-risk"""
#         # Class labels (used mainly for metrics)
#         self.y = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='class_predictions')

#
# from typing import Union, List, Callable
# import tensorflow as tf
# import layers
# import rnn_cell
# # 自定义 RNN Cell 需要自己实现或从外部导入
# # 注意：确保你使用的 rnn_cell 是兼容 TensorFlow 2.x 的，否则需要自己重写
# # import rnn_cell  # 自定义或第三方库提供的 RNN Cells
# CELL_TYPES = ['LRAN', 'RAN', 'LSTM', 'GRU']
# class CANTRIPModel(object):
#     def __init__(self,
#                  max_seq_len: int,
#                  max_snapshot_size: int,
#                  vocabulary_size: int,
#                  observation_embedding_size: int,
#                  delta_encoding_size: int,
#                  num_hidden: Union[int, List[int]],
#                  cell_type: str,
#                  batch_size: int,
#                  snapshot_encoder: Callable[['CANTRIPModel'], tf.Tensor],
#                  dropout: float = 0.,
#                  vocab_dropout: float = None,
#                  num_classes: int = 2,
#                  delta_combine: str = "concat",
#                  embed_delta: bool = False,
#                  rnn_highway_depth: int = 3,
#                  rnn_direction='forward'):
#         self.max_seq_len = max_seq_len
#         self.max_snapshot_size = max_snapshot_size
#         self.vocabulary_size = vocabulary_size
#         self.embedding_size = observation_embedding_size
#         self.num_hidden = num_hidden
#         self.batch_size = batch_size
#         self.num_classes = num_classes
#         self.delta_encoding_size = delta_encoding_size
#         self.dropout = dropout
#         self.vocab_dropout = vocab_dropout or dropout
#         self.cell_type = cell_type
#         self.delta_combine = delta_combine
#         self.embed_delta = embed_delta
#         self.rnn_highway_depth = rnn_highway_depth
#         self.rnn_direction = rnn_direction
#
#         if delta_combine == 'add' and not self.embed_delta and self.embedding_size != self.delta_encoding_size:
#             print("Cannot add delta embeddings of size %d to observation encodings of size %d, "
#                   "setting embed_delta=True" %
#                   (self.delta_encoding_size, self.embedding_size))
#             self.embed_delta = True
#
#         # Initialize model components
#         self._add_placeholders()
#         with tf.name_scope('snapshot_encoder'):
#             self.snapshot_encodings = snapshot_encoder(self)
#
#         if self.embed_delta:
#             self.delta_inputs = layers.Dense(units=self.embedding_size,
#                                              activation=None,
#                                              name='delta_embeddings')(self.deltas)
#         else:
#             self.delta_inputs = self.deltas
#
#         self._add_seq_rnn(cell_type)
#
#         self.logits = layers.Dense(units=self.num_classes,
#                                    activation=None,
#                                    name='class_logits')(self.seq_final_output)
#
#         self._add_postprocessing()
#
#     def _add_placeholders(self):
#         # Observation IDs
#         self.observations = tf.keras.Input(shape=(self.max_seq_len, self.max_snapshot_size),
#                                            dtype=tf.int32,
#                                            name="observations")
#         # Elapsed time deltas
#         self.deltas = tf.keras.Input(shape=(self.max_seq_len, self.delta_encoding_size),
#                                      dtype=tf.float32,
#                                      name="deltas")
#         # Snapshot sizes
#         self.snapshot_sizes = tf.keras.Input(shape=(self.max_seq_len,),
#                                              dtype=tf.int32,
#                                              name="snapshot_sizes")
#         # Chronology lengths
#         self.seq_lengths = tf.keras.Input(shape=(),
#                                           dtype=tf.int32,
#                                           name="seq_lengths")
#         # Label
#         self.labels = tf.keras.Input(shape=(),
#                                      dtype=tf.int32,
#                                      name="labels")
#         # Training flag (for dropout)
#         self.training = tf.keras.Input(shape=(),
#                                        dtype=tf.bool,
#                                        name="training")
#
#     def _add_seq_rnn(self, cell_type: str):
#         """Add the clinical picture inference module; implemented as an RNN."""
#         with tf.name_scope('sequence'):
#             # Add dropout on deltas
#             if self.dropout > 0:
#                 self.delta_inputs = layers.Dropout(rate=self.dropout)(self.delta_inputs, training=self.training)
#
#             # Concat observation_t and delta_t (deltas are already shifted by one)
#             if self.delta_combine == 'concat':
#                 self.x = tf.concat([self.snapshot_encodings, self.delta_inputs], axis=-1, name='rnn_input_concat')
#             elif self.delta_combine == 'add':
#                 self.x = self.snapshot_encodings + self.delta_inputs
#             else:
#                 raise ValueError("Invalid delta combination method: %s" % self.delta_combine)
#
#             # Add dropout on concatenated inputs
#             if self.dropout > 0:
#                 self.x = layers.Dropout(rate=self.dropout)(self.x, training=self.training)
#
#             # Define RNN cell types
#             _cell_types = {
#                 'RAN': lambda units: rnn_cell.RANCell(units),  # Custom implementation required
#                 'LSTM': tf.keras.layers.LSTMCell,
#                 'GRU': tf.keras.layers.GRUCell,
#                 # Add other custom cells here if needed
#             }
#
#             if cell_type not in _cell_types:
#                 raise ValueError('unsupported cell type %s' % cell_type)
#
#             self.cell_fn = _cell_types[cell_type]
#
#             if isinstance(self.num_hidden, int):
#                 self.num_hidden = [self.num_hidden]
#
#             # Create the RNN layer
#             if self.rnn_direction == 'bidirectional':
#                 rnn_layer = layers.Bidirectional(layers.RNN(self.cell_fn(self.num_hidden[0])))
#             else:
#                 rnn_layer = layers.RNN(self.cell_fn(self.num_hidden[0]))
#
#             self.seq_final_output = rnn_layer(self.x)
#
#             # Even more fun dropout
#             if self.dropout > 0:
#                 self.seq_final_output = layers.Dropout(rate=self.dropout)(self.seq_final_output, training=self.training)
#
#     def _add_postprocessing(self):
#         """Categorical arg-max prediction for disease-risk."""
#         self.y = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='class_predictions')


from typing import Union, List, Callable
import tensorflow as tf
from tensorflow.keras import layers
import rnn_cell
CELL_TYPES = ['LRAN','RAN', 'LSTM', 'GRU']

class CANTRIPModel(tf.keras.Model):
    # def __init__(
    #     self,
    #     max_seq_len: int,
    #     max_snapshot_size: int,
    #     vocabulary_size: int,
    #     observation_embedding_size: int,
    #     delta_encoding_size: int,
    #     num_hidden: Union[int, List[int]],
    #     cell_type: str,
    #     batch_size: int,
    #     snapshot_encoder: Callable[[tf.keras.Model], tf.Tensor],
    #     dropout: float = 0.0,
    #     vocab_dropout: float = None,
    #     num_classes: int = 2,
    #     delta_combine: str = "concat",
    #     embed_delta: bool = False,
    #     rnn_highway_depth: int = 3,
    #     rnn_direction: str = 'forward',
    #     training: bool = False
    #
    # ):
    #     super().__init__()
    #     self.max_seq_len = max_seq_len
    #     self.max_snapshot_size = max_snapshot_size
    #     self.vocabulary_size = vocabulary_size
    #     self.embedding_size = observation_embedding_size
    #     self.delta_encoding_size = delta_encoding_size
    #     self.num_hidden = num_hidden
    #     self.batch_size = batch_size
    #     self.num_classes = num_classes
    #     self.dropout = dropout
    #     self.vocab_dropout = vocab_dropout or dropout
    #     self.cell_type = cell_type
    #     self.delta_combine = delta_combine
    #     self.embed_delta = embed_delta
    #     self.rnn_highway_depth = rnn_highway_depth
    #     self.rnn_direction = rnn_direction
    #     self.training = training
    #     self.snapshot_encoder = snapshot_encoder(self)

    def __init__(
            self,
            max_seq_len: int,
            max_snapshot_size: int,
            vocabulary_size: int,
            observation_embedding_size: int,
            delta_encoding_size: int,
            num_hidden: Union[int, List[int]],
            cell_type: str,
            batch_size: int,
            snapshot_encoder: Callable[[tf.keras.Model, tf.Tensor], tf.Tensor],
            dropout: float = 0.0,
            vocab_dropout: float = None,
            num_classes: int = 2,
            delta_combine: str = "concat",
            embed_delta: bool = False,
            rnn_highway_depth: int = 3,
            rnn_direction: str = 'forward',
            training: bool = False,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_snapshot_size = max_snapshot_size
        self.vocabulary_size = vocabulary_size
        self.embedding_size = observation_embedding_size
        self.delta_encoding_size = delta_encoding_size
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.vocab_dropout = vocab_dropout or dropout
        self.cell_type = cell_type
        self.delta_combine = delta_combine
        self.embed_delta = embed_delta
        self.rnn_highway_depth = rnn_highway_depth
        self.rnn_direction = rnn_direction
        self.training = training

        if delta_combine == 'add' and not embed_delta and observation_embedding_size != delta_encoding_size:
            print(f"Setting embed_delta=True to match embedding sizes")
            self.embed_delta = True

        # 定义输入层
        self.observations = layers.Input(
            shape=(max_seq_len, max_snapshot_size), dtype=tf.int32, name="observations"
        )
        self.deltas = layers.Input(
            shape=(max_seq_len, delta_encoding_size), dtype=tf.float32, name="deltas"
        )
        self.snapshot_sizes = layers.Input(shape=(max_seq_len,), dtype=tf.int32, name="snapshot_sizes")
        self.seq_lengths = layers.Input(shape=(), dtype=tf.int32, name="seq_lengths")
        self.labels_input = layers.Input(shape=(), dtype=tf.int32, name="labels")
        self.training_input = layers.Input(shape=(), dtype=tf.bool, name="training")

        # Snapshot编码器
        with tf.name_scope('snapshot_encoder'):
            self.snapshot_encodings = snapshot_encoder(self)

        # Delta编码处理
        if self.embed_delta:
            self.delta_proj = layers.Dense(observation_embedding_size, name='delta_embeddings')
            delta_inputs = self.delta_proj(self.deltas)
        else:
            delta_inputs = self.deltas

        # 合并观察和Delta
        if delta_combine == 'concat':
            combined = tf.concat([self.snapshot_encodings, delta_inputs], axis=-1)
        elif delta_combine == 'add':
            combined = tf.add(self.snapshot_encodings, delta_inputs)
        else:
            raise ValueError(f"Invalid delta combine method: {delta_combine}")

        # Dropout
        if dropout > 0:
            combined = layers.Dropout(dropout)(combined, training=self.training_input)

        # 构建RNN单元
        if isinstance(num_hidden, int):
            num_hidden = [num_hidden]
        cells = []
        for units in num_hidden:
            if cell_type == 'RAN':
                cell = rnn_cell.RANCell(units)
            elif cell_type == 'LSTM':
                cell = layers.LSTMCell(units)
            elif cell_type == 'GRU':
                cell = layers.GRUCell(units)
            else:
                raise ValueError(f"Unsupported cell type: {cell_type}")
            cells.append(cell)
        rnn_cell = cells[0] if len(cells) == 1 else tf.keras.layers.StackedRNNCells(cells)

        # 双向或单向RNN
        if rnn_direction == 'bidirectional':
            self.rnn = layers.Bidirectional(layers.RNN(rnn_cell, return_sequences=False))
        else:
            self.rnn = layers.RNN(rnn_cell, return_sequences=False)
        rnn_output = self.rnn(combined)

        # 最终分类层
        if dropout > 0:
            rnn_output = layers.Dropout(dropout)(rnn_output, training=self.training_input)
        self.logits = layers.Dense(num_classes, name='class_logits')(rnn_output)
        self.predictions = tf.argmax(self.logits, axis=-1, name='predictions')

        # 构建模型
        self.build(inputs=[
            self.observations,
            self.deltas,
            self.snapshot_sizes,
            self.seq_lengths,
            self.labels_input,
            self.training_input
        ])

    def call(self, inputs, training=None):
        # 解包输入
        obs, deltas, sizes, seq_len, labels, training_flag = inputs
        # 前向传播（已在__init__中构建，此处可简化）
        return self.logits

