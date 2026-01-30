from torchgen.model import FunctionSchema
native_schema = FunctionSchema.parse("aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor")