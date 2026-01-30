import numpy as np

@test_util.run_all_in_graph_and_eager_modes
class KerasAccuracyTest(test.TestCase):

 def test_accuracy_vs_accuracy(self):
    y_true = constant_op.constant([1, 0, 1])
    y_pred = constant_op.constant([0.8, 0.1, 0.9])
    ret_a_tensor = metrics.accuracy(y_true, y_pred)
    ret_a = np.mean(self.evaluate(ret_a_tensor))
    
    acc_obj = metrics.Accuracy(name='my_acc')
    self.evaluate(variables.variables_initializer(acc_obj.variables))
    update_op = acc_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    ret_b = self.evaluate(acc_obj.result())
    self.assertEqual(ret_a, ret_b)