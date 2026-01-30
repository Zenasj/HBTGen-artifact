import torch
from models.eyenet import EyeNet

device = torch.device('cpu')

checkpoint = torch.load('./checkpoint.pt', map_location=device)

nstack = checkpoint['nstack']
nfeatures = checkpoint['nfeatures']
nlandmarks = checkpoint['nlandmarks']

model = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

input_frame = torch.rand((1, 160, 96))

exported_program: torch.export.ExportedProgram= torch.export.export(model, args=(input_frame,), strict=False)