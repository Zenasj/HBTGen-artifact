import torch

def test_closure_out_of_scope_cell_with_cond(self):
        from functorch.experimental.control_flow import cond
        cell1 = torch.rand(3, 3)
        cell2 = torch.rand(3, 3)
        orig3 = torch.rand(3, 3)
        def test(x):
            cell3 = orig3.clone()
            def then():
                nonlocal cell3
                cell3 += cell1
                return cell3
            def els():
                nonlocal cell3
                cell3 += cell2
                return cell3
            return cond(x > 0, then, els, [])
        opt_fn = torch._dynamo.optimize("eager")(test)
        result1 = opt_fn(1)
        self.assertTrue(torch.allclose(result1, orig3 + cell1))
        result2 = opt_fn(-1)
        self.assertTrue(torch.allclose(result1, orig3 + cell1 + cell2))