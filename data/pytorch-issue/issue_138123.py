import torch

model_type = "DPT_Large"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = torch.hub.load("intel-isl/MiDaS", model_type)
model.eval()

input_frame = torch.randn(1,3,384,480)

exported_program: torch.export.ExportedProgram= torch.export.export(model, args=(input_frame,), strict=False)