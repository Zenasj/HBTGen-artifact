import torch
import random

# Really annoying intersection of specialization and RandomValueSource
    # If we get a RandomValueSource with a single element tensor, we should return a ConstantVariable like other
    # unspects... but if we do, we break the bytecode assumptions and guards will not work as we will be referring
    # to a name from a source that is not there. If we call .item() and take the wrapped_value out, where we do
    # wrapped_value = wrapped_value.item() where we send unspec down to wrap_fx_proxy, this test passes and then
    # some models fail on missing codegen.tx.output.random_values_var. If we let the tensor value go into wrap as
    # it is, this test fails.
    # The real solution here is to rewrite RandomValueSource and all the codegen it does from the ground up.

def test_rand_new(self):
        @torch.compile(backend="eager")
        def core():
            idx_size = [10]
            idx_size[random.randint(0, 1)] = random.randint(1, 9)
            t= tuple(idx_size)

            # If I remove this line  or the line after I get IndexError: list assignment index out of range
            # instread of
            src_size = [random.randint(1, 5) + s for s in idx_size]
            idx = torch.empty(t)
        core()
        core()

idx_size[random.randint(0, 1)] = random.randint(1, 9)

if len(tx.random_calls) > 0:
            append_prefix_insts()
            random_calls_instructions = []
            self.random_values_var = self.new_var("random_values")
            rand_fn = disable(_get_gen_rand_values_fn(tx.random_calls))
            rand_fn_name = self.install_global("__gen_rand_values", rand_fn)
            codegen = PyCodegen(tx, root)
            random_calls_instructions.extend(
                codegen.load_function_name(rand_fn_name, True)
            )
            random_calls_instructions.extend(create_call_function(0, False))
            random_calls_instructions.append(
                codegen.create_store(tx.output.random_values_var),
            )
            self.add_output_instructions(random_calls_instructions)

# Cleanup the outputGraph to delete the held tensors. We perform the
                # cleanup only for InstructionTranslator and not
                # InliningInstructionTranslator. The InliningInstructionTranslator
                # mutates the output object and is restored to original state if
                # there was an exception.