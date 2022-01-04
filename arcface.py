import tensorflow as tf
import math

bn_mom = 0.9

class ArcFaceLayer(tf.keras.layers.Layer):
    # Input:  Embeddings, One-hot Labels
    # Output: Softmax of Logits
    def __init__(self, output_dim, s=64., margin=0.5, arccos=False, easy_margin=False, **kwargs):
        assert s > 0.0, "Argument 's' must be greater than 0."
        assert margin >= 0.0, "Argument 'margin' must be greater than or equal to 0."
        assert margin < (math.pi / 2), "Argument 'margin' must be less than PI/2."
        super(ArcFaceLayer, self).__init__()
        self.output_dim = output_dim
        self.s = s
        self.margin = margin
        self.cos_margin = tf.math.cos(margin)
        self.sin_margin = tf.math.sin(margin)
        self.threshold = tf.math.cos(math.pi - margin)
        self.mm = tf.math.sin(math.pi - margin) * margin
        self.arccos = arccos            # True: arccos,      False: trigonometric identities
        self.easy_margin = easy_margin  # True: easy margin, False: normal margin

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.output_dim),
                                 initializer=tf.keras.initializers.RandomNormal(0.0, 0.01),
                                 trainable=True,
                                 name='kernel')
        super(ArcFaceLayer, self).build(input_shape)
        
    def call(self, embedding, **kwargs):
        kwargs.setdefault('labels', None)
        kwargs.setdefault('training', False)
        norm_weights = tf.norm(self.w, axis=0)
        norm_embedding = tf.expand_dims(tf.norm(embedding, axis=-1), axis=-1)
        logits = tf.clip_by_value(tf.matmul(embedding, self.w) / norm_weights / norm_embedding, -1.0, 1.0)
        
        if not kwargs['training']:
            logits = logits * self.s

        elif kwargs['training'] and kwargs['labels'] is not None:
            labels = kwargs['labels']
            if len(labels.shape) == 1:
                one_hot_labels = tf.one_hot(labels, depth=self.output_dim) # shape = [batch_size, class_dim]
            else:
                assert len(labels.shape) == 2, "The dimension of labels is too large."
                one_hot_labels = labels # shape = [batch_size, class_dim]
            one_hot_boolean_mask = tf.equal(one_hot_labels, 1.)
            cos_t = tf.boolean_mask(logits, one_hot_boolean_mask)

            if self.arccos:
                t = tf.math.acos(cos_t)
                t += self.margin
                margined_cos_t = tf.math.cos(t)
            else:
                sin_t = tf.math.sqrt(1 - tf.math.square(cos_t) + 2e-5)
                margined_cos_t = tf.math.multiply(cos_t, self.cos_margin) - tf.math.multiply(sin_t, self.sin_margin)

            if self.easy_margin:
                new_cos_t = tf.where(cos_t > 0, margined_cos_t, cos_t)
                logits = tf.where(one_hot_boolean_mask, tf.expand_dims(new_cos_t, -1), logits) * self.s
            else:
                new_cos_t = tf.where(cos_t > self.threshold, margined_cos_t, cos_t - self.mm)
                logits = tf.where(one_hot_boolean_mask, tf.expand_dims(new_cos_t, -1), logits) * self.s
        else:
            assert False, "Argument 'labels' is necessary when training=True."
        return tf.nn.softmax(logits)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            's': self.s,
            'margin': self.margin,
            'output_dim': self.output_dim,
            'arccos': self.arccos,
            'easy_margin': self.easy_margin
        })
        return config

def get_fc1(last_conv, emb_size):
    x = tf.keras.layers.BatchNormalization(momentum=bn_mom,
                                           name='bn1')(last_conv)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(emb_size, name='pre_fc1')(x)
    x = tf.keras.layers.BatchNormalization(momentum=bn_mom,
                                           name='fc1')(x)
    return x

def residual_unit_v3(input_data, num_filter, stride, dim_match, name):
    """
    Return ResNet Unit for building ResNet
    Parameters
    ----------
    input : 
        Input data
    num_filter : int
        Number of output channels
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    """
    bn1 = tf.keras.layers.BatchNormalization(momentum=bn_mom,
                                           epsilon=2e-5,
                                           name=name + '_bn1')(input_data)
    pad1 = tf.keras.layers.ZeroPadding2D(
        padding=(1, 1), name=name + '_conv1_pad')(bn1)
    conv1 = tf.keras.layers.Conv2D(num_filter, (3, 3),
                               strides=(1, 1),
                               padding='valid',
                               use_bias=False,
                               name=name + '_conv1')(pad1)
    bn2 = tf.keras.layers.BatchNormalization(momentum=bn_mom,
                                           epsilon=2e-5,
                                           name=name + '_bn2')(conv1)
    act1 = tf.keras.layers.PReLU(name=name + '_relu1')(bn2)
    pad2 = tf.keras.layers.ZeroPadding2D(
        padding=(1, 1), name=name + '_conv2_pad')(act1)
    conv2 = tf.keras.layers.Conv2D(num_filter, (3, 3),
                               strides=stride,
                               padding='valid',
                               use_bias=False,
                               name=name + '_conv2')(pad2)
    bn3 = tf.keras.layers.BatchNormalization(momentum=bn_mom,
                                           epsilon=2e-5,
                                           name=name + '_bn3')(conv2)
    if dim_match:
        shortcut = input_data
    else:
        conv1sc = tf.keras.layers.Conv2D(num_filter, (1, 1),
                               strides=stride,
                               padding='valid',
                               use_bias=False,
                               name=name + '_conv1sc')(input_data)
        shortcut = tf.keras.layers.BatchNormalization(momentum=bn_mom,
                                           epsilon=2e-5,
                                           name=name + '_sc')(conv1sc)
    return bn3 + shortcut

def ResNet50(emb_size = 512):
    # Input preprocessing:
    # input ∈ [0, 255]
    # input -= 127.5
    # input /= 128
    input_shape = [112, 112, 3]
    filter_list = [64, 64, 128, 256, 512]
    units = [3, 4, 14, 3]
    num_stages = 4

    img_input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.ZeroPadding2D(
        padding=(1, 1), name='conv0_pad')(img_input)
    x = tf.keras.layers.Conv2D(filter_list[0], (3, 3),
                               strides=(1, 1),
                               padding='valid',
                               use_bias=False,
                               name='conv0')(x)
    x = tf.keras.layers.BatchNormalization(momentum=bn_mom,
                                           epsilon=2e-5,
                                           name='bn0')(x)
    x = tf.keras.layers.PReLU(name='relu0')(x)

    for i in range(num_stages):
        x = residual_unit_v3(x, filter_list[i + 1], (2, 2), False,
                        name='stage%d_unit%d' % (i + 1, 1))
        for j in range(units[i] - 1):
            x = residual_unit_v3(x, filter_list[i + 1], (1, 1), True,
                            name='stage%d_unit%d' % (i + 1, j + 2))
    fc1 = get_fc1(x, emb_size)

    model = tf.keras.models.Model(img_input, fc1, name='resnet50')
    model.trainable = True
    for layer in model.layers:
        layer.trainable = True
    return model