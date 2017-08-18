def unet2d(window_shape=(128, 128), nb_filters_base=32, conv_kernel_init='he_normal',
         prop_dropout_base=0.25, upsampling_or_transpose='transpose'):
    """Builds and returns the UNet architecture using Keras.

    # Arguments
        window_shape: tuple of two equivalent integers defining the input/output window shape.
        nb_filters_base: number of convolutional filters used at the first layer. This is doubled
            after every pooling layer, four times until the bottleneck layer, and then it gets
            divided by two four times to the output layer.
        conv_kernel_init: weight initialization for the convolutional kernels. He initialization
            is considered best-practice when using ReLU activations, as is the case in this network.
        prop_dropout_base: proportion of dropout after the first pooling layer. Two-times the
            proportion is used after subsequent pooling layers on the downward pass.
        upsampling_or_transpose: whether to use Upsampling2D or Conv2DTranspose layers on the upward
            pass. The original paper used Conv2DTranspose ("Deconvolution").

    # Returns
        model: Keras model, not compiled.

    """

    from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate, BatchNormalization, Lambda, UpSampling2D, Activation
    from keras.models import Model


    drp = prop_dropout_base
    nfb = nb_filters_base
    cki = conv_kernel_init

    # Theano vs. TF setup.
    assert K.backend() == 'tensorflow', 'Theano implementation is incomplete.'

    def up_layer(nb_filters, x):
        if upsampling_or_transpose == 'transpose':
            x = Conv2DTranspose(nb_filters, 2, strides=2,
                                kernel_initializer=cki)(x)
            x = BatchNormalization(momentum=0.5)(x)
            return Activation('relu')(x)
        else:
            return UpSampling2D()(x)

    def conv_layer(nb_filters, x):
        x = Conv2D(nb_filters, (3, 3), strides=(1, 1),
                   padding='same', kernel_initializer=cki)(x)
        x = BatchNormalization(axis=-1)(x)
        return Activation('relu')(x)

    x = inputs = Input(window_shape)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)

    x = conv_layer(nfb, x)
    x = conv_layer(nfb, x)
    dc_0_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 2, x)
    x = conv_layer(nfb * 2, x)
    x = Dropout(drp)(x)
    dc_1_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 4, x)
    x = conv_layer(nfb * 4, x)
    x = Dropout(drp * 2)(x)
    dc_2_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 8, x)
    x = conv_layer(nfb * 8, x)
    x = Dropout(drp * 2)(x)
    dc_3_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 16, x)
    x = conv_layer(nfb * 16, x)
    x = up_layer(nfb * 8, x)
    x = Dropout(drp * 2)(x)

    x = concatenate([x, dc_3_out], axis=-1)
    x = conv_layer(nfb * 8, x)
    x = conv_layer(nfb * 8, x)
    x = up_layer(nfb * 4, x)
    x = Dropout(drp * 2)(x)

    x = concatenate([x, dc_2_out], axis=-1)
    x = conv_layer(nfb * 4, x)
    x = conv_layer(nfb * 4, x)
    x = up_layer(nfb * 2, x)
    x = Dropout(drp * 2)(x)

    x = concatenate([x, dc_1_out], axis=-1)
    x = conv_layer(nfb * 2, x)
    x = conv_layer(nfb * 2, x)
    x = up_layer(nfb, x)
    x = Dropout(drp)(x)

    x = concatenate([x, dc_0_out], axis=-1)
    x = conv_layer(nfb, x)
    x = conv_layer(nfb, x)
    x = Conv2D(2, 1, activation='softmax')(x)
    x = Lambda(lambda x: x[:, :, :, -1])(x)
    model = Model(inputs=inputs, outputs=x)
    return model
