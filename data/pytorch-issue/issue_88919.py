import torch

if isinstance(window, str):
        windows = {
            "hann": torch.hann_window,
            "hamming": torch.hamming_window,
            "blackman": torch.blackman_window
        }
        if window not in windows.keys():
            raise ValueError(f"""{window} is not a valid window name.
            			Available windows are {windows.keys()}""")
        
        else:
            win_length = n_fft if win_length is None else win_length
            window = windows[window](win_length, device=input.device)