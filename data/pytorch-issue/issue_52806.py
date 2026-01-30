dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=min(8, cast(int, os.cpu_count())),
    pin_memory=False,
    drop_last=False,
    prefetch_factor=1,
    #collate_fn=default_collate,
)