import torch.nn as nn

if __name__=="__main__":
    import torch
    model = torch.nn.TransformerEncoderLayer(
        d_model=256,
        nhead=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="gelu",
        batch_first=True,
    )

    model = torch.compile(
        model,
        dynamic=True,
        fullgraph=True,
    )

    device = 'cuda'
    model = model.cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    )

    model.train()

    for i in range(10):
        with torch.cuda.amp.autocast(enabled=False):
            optimizer.zero_grad()
            x = torch.randn(
                10,
                2 + i * 2,
                256,
            ).cuda()
            memory_mask = torch.zeros(
                x.size(0),
                x.size(1),
                device=x.device,
            ).bool()
            res = model(
                x,
                src_key_padding_mask=memory_mask,
            )
            loss = ((res - x) ** 2).mean()

        loss.backward()
        optimizer.step()

        print(i)