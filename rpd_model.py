import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.drop = tf.keras.layers.Dropout(kwargs['dropout'])

    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        x = self.pool(x)
        x = self.drop(x)

        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


def RPD_Model_1(n_classes=1, input_shape=(224, 224, 3), weights='imagenet',
              num_heads=12, key_dim=128, dropout=0.2, name='RPD_Model'):
    return tf.keras.Sequential([
        tf.keras.applications.InceptionV3(include_top=False, weights=weights, input_shape=input_shape),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Reshape((128, 16)),
        CausalSelfAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(128, activation='relu', name='cfp_dense'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(n_classes, name='cfp_predictions'),
    ], name=name)


class RPD_Model_2(tf.keras.Model):
    def __init__(self, n_classes=1, input_shape=(224, 224, 3), weights='imagenet',
                 num_heads=12, key_dim=128, dropout=0.2, name='RPD_Model', **kwargs):
        super().__init__(name=name, **kwargs)

        def create_branch(branch_name):
            return tf.keras.Sequential([
                tf.keras.applications.InceptionV3(include_top=False, weights=weights, input_shape=input_shape),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Reshape((128, 16)),
                CausalSelfAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
            ], name=branch_name)

        self.faf_branch = create_branch(branch_name='faf_branch')
        self.cfp_branch = create_branch(branch_name='cfp_branch')

        self.faf_to_cfp = CrossAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout, name='faf_to_cfp')
        self.cfp_to_faf = CrossAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout, name='cfp_to_faf')

        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.combined_dense = tf.keras.layers.Dense(128, activation='relu')
        self.combined_dropout = tf.keras.layers.Dropout(dropout)
        self.combined_predictions = tf.keras.layers.Dense(n_classes)

    def call(self, inputs, **kwargs):

        faf = self.faf_branch(inputs['faf_input'])
        cfp = self.cfp_branch(inputs['faf_input'])

        faf_to_cfp = self.faf_to_cfp(faf, cfp)
        cfp_to_faf = self.cfp_to_faf(cfp, faf)

        x = self.concat([faf_to_cfp, cfp_to_faf])
        x = self.combined_dense(x)
        x = self.combined_dropout(x)
        x = self.combined_predictions(x)

        return x


if __name__ == '__main__':

    faf_input = tf.random.uniform(shape=(1, 224, 224, 3), minval=1, maxval=255, dtype=tf.int32)
    cfp_input = tf.random.uniform(shape=(1, 224, 224, 3), minval=1, maxval=255, dtype=tf.int32)

    model = RPD_Model_2()
    output = model(faf_input, cfp_input)

    a=1
