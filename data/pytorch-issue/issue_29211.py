import torch


def test_fp16_categorical():
    logits_fp16 = torch.randn(20).cuda().half()

    # These are fine
    torch.argmax(logits_fp16)
    torch.max(logits_fp16)

    # This is also fine
    logits_fp32 = logits_fp16.float()
    sample = torch.distributions.Categorical(logits=logits_fp32).sample()
    print(sample)

    # This fails
    sample = torch.distributions.Categorical(logits=logits_fp16).sample()
    print(sample)


if __name__ == "__main__":
    test_fp16_categorical()