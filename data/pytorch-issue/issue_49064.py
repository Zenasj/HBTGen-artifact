import warnings
import logging
warnings.simplefilter("default")
logging.captureWarnings(True)
logging.basicConfig()

exec(r'''
convolution_notes = \
    {"groups_note": """* :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\\frac{\\text{out\_channels}}{\\text{in\_channels}}`).""",  # noqa: W605

        "depthwise_separable_note": """When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also known as a "depthwise convolution".

        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
        :math:`(C_\\text{in}=C_\\text{in}, C_\\text{out}=C_\\text{in} \\times \\text{K}, ..., \\text{groups}=C_\\text{in})`."""}  # noqa: W605,B950
''')