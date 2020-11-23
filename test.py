#     self._m1 = self.module_1
#
#
# def module_1(self):
#     def module(x):
#         x = layers.DepthwiseConv2D(self._block_args.kernel_size,
#                strides=self._block_args.strides,
#                padding='same',
#                use_bias=False,
#                depthwise_initializer=CONV_KERNEL_INITIALIZER)(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation(self._activation)(x)
#         return x
#     return module
# @classmethod
# def module_1(cls, strides, kernel_size, activation):
#     return Sequential([
#         layers.DepthwiseConv2D(kernel_size,
#                                strides=strides,
#                                padding='same',
#                                use_bias=False,
#                                depthwise_initializer=CONV_KERNEL_INITIALIZER),
#         layers.BatchNormalization(),
#
#     ])
#
# @classmethod
# def module_2(cls, strides, kernel_size, activation):
#     return Sequential([
#         cls.module_1(strides, kernel_size, activation),
#         cls.module_1(strides, kernel_size, activation)
#     ])
#
# @classmethod
# def module_3(cls, filters, activation):
#     return Sequential([
#         layers.GlobalAveragePooling2D(),
#         layers.Reshape(target_shape=(1, 1, filters)),
#         layers.Conv2D(
#             filters=filters/2,
#             activation=activation,
#             kernel_size=1,
#             padding="same",
#             use_bias=True,
#             kernel_initializer=CONV_KERNEL_INITIALIZER,
#         ),
#         layers.Conv2D(
#             filters=filters,
#             activation="sigmoid",
#             kernel_size=1,
#             padding="same",
#             use_bias=True,
#             kernel_initializer=CONV_KERNEL_INITIALIZER,
#         )
#     ])
#
# def module_4(self, filters):
#     return Sequential([
#         layers.multiply([self.outputs[0] , self.outputs[-1]]),
#         layers.Conv2D(
#             filters=filters,
#             kernel_size=1,
#             padding="same",
#             use_bias=False,
#             kernel_initializer=CONV_KERNEL_INITIALIZER,
#         ),
#         layers.BatchNormalization()
#     ])
#
#
# @classmethod
# def module_5(cls, filters, dropout_rate):
#     return Sequential([
#         cls.module_4(filters),
#         layers.Dropout(dropout_rate)
#     ])

# def call(self, x, training=None, mask=None):
#     x = self._stem(x)
#     x = self._m1(x)
#     # x = self._m2(x)
#     # x = self._m3(x)
#     # x = self._m4(x)
#     # x = self._m5(x)
#     x = self._top(x)
#     x = self._classifier(x)
#     return x
