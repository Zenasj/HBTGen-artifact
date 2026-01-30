import unittest 
import torch

class MyTestCase(unittest.TestCase): 
  
   def test_1(self):
      t = torch.tensor((1 + 1j), device='cpu', dtype=torch.complex128) 
      with self.assertRaises(RuntimeError): 
         torch.max(t, dim=0)
  
if __name__ == '__main__':  
    unittest.main()