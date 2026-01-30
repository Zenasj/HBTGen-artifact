import random
import tensorflow as tf


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.count = 0
        self.images = []
        print(tf.autograph.to_code(self.query))

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            # if the buffer is not full; keep inserting current images to the buffer
            if self.count < self.pool_size:
                self.count = self.count + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    # by 50% chance, the buffer will return a previously stored image
                    # and insert the current image into the buffer
                    # randint is inclusive
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return return_images

def tf__query(self, images):
  do_return = False
  retval_ = ag__.UndefinedReturnValue()
  cond_2 = self.pool_size == 0

  def get_state_2():
    return self.count, self.images[random_id]

  def set_state_2(vals):
    self.count, self.images[random_id] = vals

  def if_true_2():
    do_return = True
    retval_ = images
    return retval_

  def if_false_2():
    return_images = []

    def loop_body(loop_vars, self_count):
      image = loop_vars
      cond_1 = self_count < self.pool_size

      def get_state_1():
        return self_count, self.images[random_id]

      def set_state_1(vals):
        self_count, self.images[random_id] = vals

      def if_true_1():
        self_count = self_count + 1
        ag__.converted_call('append', self.images, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (image,), None)
        ag__.converted_call('append', return_images, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (image,), None)
        return ag__.match_staging_level(1, cond_1)

      def if_false_1():
        p = ag__.converted_call('random', random, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (), None)
        cond = p > 0.5

        def get_state():
          return self.images[random_id],

        def set_state(vals):
          self.images[random_id], = vals

        def if_true():
          random_id = ag__.converted_call('randint', random, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (0, self.pool_size - 1), None)
          tmp = ag__.converted_call('clone', self.images[random_id], ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (), None)
          self.images[random_id] = image
          ag__.converted_call('append', return_images, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (tmp,), None)
          return ag__.match_staging_level(1, cond)

        def if_false():
          ag__.converted_call('append', return_images, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (image,), None)
          return ag__.match_staging_level(1, cond)
        ag__.if_stmt(cond, if_true, if_false, get_state, set_state)
        return ag__.match_staging_level(1, cond_1)
      ag__.if_stmt(cond_1, if_true_1, if_false_1, get_state_1, set_state_1)
      return self_count,
    self.count, = ag__.for_stmt(images, None, loop_body, (self.count,))
    do_return = True
    retval_ = return_images
    return retval_
  retval_ = ag__.if_stmt(cond_2, if_true_2, if_false_2, get_state_2, set_state_2)
  cond_3 = ag__.is_undefined_return(retval_)

  def get_state_3():
    return ()

  def set_state_3(_):
    pass

  def if_true_3():
    retval_ = None
    return retval_

  def if_false_3():
    return retval_
  retval_ = ag__.if_stmt(cond_3, if_true_3, if_false_3, get_state_3, set_state_3)
  return retval_

def tf__query(self, images):
  do_return = False
  retval_ = ag__.UndefinedReturnValue()
  cond_2 = self.pool_size == 0

  def get_state_2():
    return self.count, self.images[random_id]

  def set_state_2(vals):
    self.count, self.images[random_id] = vals

  def if_true_2():
    do_return = True
    retval_ = images
    return retval_

  def if_false_2():
    return_images = []

    def loop_body(loop_vars, self_count):
      image = loop_vars
      cond_1 = self_count < self.pool_size

      def get_state_1():
        return self_count, self.images[random_id]

      def set_state_1(vals):
        self_count, self.images[random_id] = vals

      def if_true_1():
        self_count = self_count + 1
        ag__.converted_call('append', self.images, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (image,), None)
        ag__.converted_call('append', return_images, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (image,), None)
        return ag__.match_staging_level(1, cond_1)

      def if_false_1():
        p = ag__.converted_call('random', random, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (), None)
        cond = p > 0.5

        def get_state():
          return self.images[random_id],

        def set_state(vals):
          self.images[random_id], = vals

        def if_true():
          random_id = ag__.converted_call('randint', random, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (0, self.pool_size - 1), None)
          tmp = ag__.converted_call('clone', self.images[random_id], ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (), None)
          self.images[random_id] = image
          ag__.converted_call('append', return_images, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (tmp,), None)
          return ag__.match_staging_level(1, cond)

        def if_false():
          ag__.converted_call('append', return_images, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (image,), None)
          return ag__.match_staging_level(1, cond)
        ag__.if_stmt(cond, if_true, if_false, get_state, set_state)
        return ag__.match_staging_level(1, cond_1)
      ag__.if_stmt(cond_1, if_true_1, if_false_1, get_state_1, set_state_1)
      return self_count,
    self.count, = ag__.for_stmt(images, None, loop_body, (self.count,))
    do_return = True
    retval_ = return_images
    return retval_
  retval_ = ag__.if_stmt(cond_2, if_true_2, if_false_2, get_state_2, set_state_2)
  cond_3 = ag__.is_undefined_return(retval_)

  def get_state_3():
    return ()

  def set_state_3(_):
    pass

  def if_true_3():
    retval_ = None
    return retval_

  def if_false_3():
    return retval_
  retval_ = ag__.if_stmt(cond_3, if_true_3, if_false_3, get_state_3, set_state_3)
  return retval_

def get_state():
          return self.images[random_id],