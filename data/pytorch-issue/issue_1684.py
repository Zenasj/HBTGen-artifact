def pixel_shuffle(input, scale_factor):
    batch_size, in_channels, in_height, in_width = input.size()

    out_channels = channels // (scale_factor * scale_factor)
    out_height = in_height * scale_factor
    out_width = in_width * scale_factor

    if scale_factor >= 1:
        input_view = input.contiguous().view(
            batch_size, channels, upscale_factor, upscale_factor,
            in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = 1 / scale_factor
        input_view = input.contiguous().view(
            batch_size, channels, out_height, block_size,
            out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)