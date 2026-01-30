from tensorflow.python.platform import tf_logging
import logging
import sys
tf_logging.set_verbosity(tf_logging.INFO)
logger = logging.getLogger("tensorflow")
if len(logger.handlers) == 1:
  logger.handlers = []
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter(
      "%(asctime)s - [%(filename)s:%(lineno)d] - %(name)s - %(levelname)s - %(message)s")
  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  ch.setFormatter(formatter)
  logger.addHandler(ch)

class PipelineTest(test.TestCase):
  def __init__(self, method_name="runTest"):
    super(PipelineTest, self).__init__(method_name)

  def testBuild(self):
      tf_logging.info("Output")