import torch

def test_win_length_padding1():
    input1 = torch.cat((torch.ones(300), torch.zeros(618)))
    input2 = torch.cat((torch.ones(400), torch.zeros(518)))

    out1 = torch.stft(input1, n_fft=512, win_length=300, hop_length=1024, center=False).squeeze_()
    out2 = torch.stft(input2, n_fft=512, win_length=300, hop_length=1024, center=False).squeeze_()

    out1 = out1[:,0]*out1[:,0]+out1[:,1]*out1[:,1]
    out2 = out2[:,0]*out2[:,0]+out2[:,1]*out2[:,1]

    assert torch.isclose(out1, out2).all()


def test_win_length_padding2():
    input1 = torch.cat((torch.zeros(106), torch.ones(300), torch.zeros(618)))
    input2 = torch.cat((torch.ones(406), torch.zeros(618)))

    out1 = torch.stft(input1, n_fft=512, win_length=300, hop_length=1024, center=False).squeeze_()
    out2 = torch.stft(input2, n_fft=512, win_length=300, hop_length=1024, center=False).squeeze_()

    out1 = out1[:,0]*out1[:,0]+out1[:,1]*out1[:,1]
    out2 = out2[:,0]*out2[:,0]+out2[:,1]*out2[:,1]

    assert torch.isclose(out1, out2).all()