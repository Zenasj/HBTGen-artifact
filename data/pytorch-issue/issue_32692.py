try:
    param.copy_(input_param)
except Exception:
    error_msgs.append('While copying the parameter named "{}", '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(key, param.size(), input_param.size()))

try:
    param.copy_(input_param)
except Exception as ex:
    error_msgs.append('While copying the parameter named "{}", '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      'an exception occured: {}'
                      .format(key, param.size(), input_param.size(), ex.args))