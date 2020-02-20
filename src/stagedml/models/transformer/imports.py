
from functools import partial
from tensorflow import ( Tensor, random_normal_initializer )
from tensorflow.keras.layers import ( Layer, LayerNormalization, Dense )
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from official.nlp.bert_modeling import Dense3D
from official.nlp.transformer.model_utils import ( get_padding_bias,
    get_padding, get_position_encoding, get_decoder_self_attention_bias )
from official.nlp.transformer.metrics import ( padded_accuracy,
    padded_accuracy_topk, padded_accuracy_top5, padded_neg_log_perplexity,
    padded_sequence_accuracy, transformer_loss )
from official.nlp.transformer.beam_search import sequence_beam_search
from official.nlp.transformer.utils.tokenizer import EOS_ID
from official.nlp.transformer.optimizer import ( LearningRateSchedule,
    LearningRateScheduler, LearningRateFn )
from official.nlp.transformer.data_pipeline import ( train_input_fn,
    map_data_for_transformer_fn )
from official.nlp.transformer.model_params import BASE_PARAMS
