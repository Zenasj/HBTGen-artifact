import torch.nn as nn

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import tqdm

from oread import datasets
from oread.models import stam, timesformer, vivit, x3d


def run_inference(
    chkpt_path,
    data_filename,
    root,
    num_classes,
    mean,
    std,
    split="inference",
    modelname="x3d",
    device="default",
    datapoint_loc_label="filepath",
    batch_size=8,
    frames=64,
    task="class",
    save_to_file="view_pred.csv",
    save_freq=20,
    append_to_previous=True,
    apply_mask=False,
    transforms=[],
    resize=224,
    target_label="Outcome",
):
    """Runs test epoch, computes metrics, and plots test set results.

    Args:
        data_filename ([str]): [name of csv to load]
        num_classes ([str]): [number of classes to predict]
        chkpt_path ([str]): [path to folder containing "best.pt"]
        modelname (str, optional): [name of model architecture to use]. Defaults to 'x3d'.
        datapoint_loc_label (str, optional): [name of column in csv with paths to data]. Defaults to 'filepath'.
        batch_size (int, optional): [batch size]. Defaults to 8.
        frames (int, optional): [number of frames to use per video]. Defaults to 64.
    """

    # This mean and STD is for CathEF ONLY

    ####mean = [120.953316, 120.953316, 120.953316]
    ####std = [39.573166, 39.573166, 39.573166]

    kwargs = {
        "num_classes": num_classes,
        "mean": mean,
        "std": std,
        "length": frames,
        "period": 1,
        "data_filename": data_filename,
        "root": root,
        "datapoint_loc_label": datapoint_loc_label,
        "apply_mask": apply_mask,
        "video_transforms": transforms,
        "resize": resize,
        "target_label": target_label,
    }

    if device == "default":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # One-hot encode target, detect number of classes
    dataset = pd.read_csv(os.path.join(data_filename), sep="µ")

    # build model, load state_dict
    # specify model
    if modelname == "x3d":
        model = x3d(num_classes=num_classes, task=task)
    if modelname == "timesformer":
        model = timesformer(num_classes=num_classes)
    if modelname == "stam":
        model = stam(num_classes=num_classes)
    elif modelname == "vivit":
        model = vivit(num_classes=num_classes)
    if modelname in [
        "c2d_r50",
        "i3d_r50",
        "slow_r50",
        "slowfast_r50",
        "slowfast_r101",
        "slowfast_16x8_r101_50_50",
        "csn_r101",
        "r2plus1d_r50",
        "x3d_xs",
        "x3d_s",
        "x3d_m",
        "x3d_l",
        "mvit_base_16x4",
        "mvit_base_32x3",
        "efficient_x3d_xs",
        "efficient_x3d_s",
    ]:
        from oread.models import pytorchvideo_model

        model = pytorchvideo_model(modelname, num_classes)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    ### If PyTorch 2.0 is used, the following line is needed to load the model
    # model = torch.compile(model)

    model.to(device)

    if device == torch.device("cuda"):
        print("Loading checkpoint from: ", os.path.join(chkpt_path, "best.pt"))
        checkpoint = torch.load(os.path.join(chkpt_path, "best.pt"))
    else:
        checkpoint = torch.load(
            os.path.join(chkpt_path, "best.pt"), map_location=torch.device("cpu")
        )

    # check if checkpoint was created with torch.nn.parallel
    key, value = list(checkpoint["state_dict"].items())[0]

    is_parallel = "module." in key
    is_parallel

    print("Is_parallel: ", is_parallel)

    # if created with torch.nn.parallel and we are using cpu, we want to remove the "module." from key names
    if device == torch.device("cpu") and is_parallel:
        state_dict = torch.load(
            os.path.join(chkpt_path, "best.pt"), map_location=torch.device("cpu")
        )["state_dict"]
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
    else:
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = torch.load(os.path.join(chkpt_path, "best.pt"))["state_dict"]

            new_state_dict = OrderedDict()
            print("Keys don't match, removing _orig_mod.")
            for k, v in state_dict.items():
                name = k[10:]  # remove `_orig_mod.`
                new_state_dict[name] = v

    model.eval()
    # pdb.set_trace()
    dataset = datasets.Echo_Inference(split=split, **kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=15,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    print("Starting inference epoch...")
    run_inference_epoch(
        model,
        modelname,
        dataloader,
        "inference",
        device=device,
        save_to_file=save_to_file,
        save_freq=save_freq,
        root=root,
        append_to_previous=append_to_previous,
        num_classes=num_classes,
    )


def run_inference_epoch(
    model,
    modelname,
    dataloader,
    phase,
    device,
    blocks=None,
    save_to_file="view_pred.csv",
    save_freq=20,
    append_to_previous=False,
    root="",
    num_classes=21,
):
    yhat_for_save = []
    fns_for_save = []
    save_count = 0
    nbatch = len(dataloader)
    preds_df = None

    with torch.set_grad_enabled(False):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (i, (X, filenames)) in enumerate(dataloader):
                X = X.to(device)
                average = len(X.shape) == 6
                if average:
                    batch, n_crops, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)

                if modelname in ["vivit", "timesformer", "stam"]:
                    X = X.permute(0, 2, 1, 3, 4)
                if blocks is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat(
                        [model(X[j : (j + blocks), ...]) for j in range(0, X.shape[0], blocks)]
                    )

                if average:
                    outputs = outputs.view(batch, n_crops, -1).mean(1)
                if modelname in [
                    "c2d_r50",
                    "i3d_r50",
                    "slow_r50",
                    "slowfast_r50",
                    "slowfast_r101",
                    "slowfast_16x8_r101_50_50",
                    "csn_r101",
                    "r2plus1d_r50",
                    "x3d_xs",
                    "x3d_s",
                    "x3d_m",
                    "x3d_l",
                    "mvit_base_16x4",
                    "mvit_base_32x3",
                    "efficient_x3d_xs",
                    "efficient_x3d_s",
                ]:
                    outputs = outputs[:, 1].float()
                elif outputs.shape[0] == 1:
                    reshaped = outputs.squeeze(2)
                    print("outputs", outputs)
                    print("reshaped", reshaped)
                else:
                    reshaped = outputs.squeeze()
                    print("outputs", outputs)
                    print("reshaped squeeze", reshaped)
                

                yhat_for_save.append(outputs.squeeze().to("cpu").detach().numpy())
                fns_for_save.append(filenames)
                # pdb.set_trace()

                if (i + 1) % save_freq == 0 or i == nbatch - 1:
                    # pdb.set_trace()
                    flatten_yhat_for_save = [
                        item for sublist in yhat_for_save for item in sublist
                    ]

                    if num_classes == 1:
                        # pdb.set_trace()
                        this_preds = pd.DataFrame(flatten_yhat_for_save)
                    elif num_classes == 2:
                        this_preds = pd.DataFrame(np.hstack(yhat_for_save))
                    else:
                        this_preds = pd.DataFrame(np.row_stack(yhat_for_save))

                    flattened_list = [item for sublist in fns_for_save for item in sublist]
                    this_preds["filename"] = flattened_list

                    # Assuming this_preds is your DataFrame
                    integer_columns = [
                        col for col in this_preds.columns if isinstance(col, int) or col.isdigit()
                    ]

                    # Select the columns with integer names
                    selected_columns = this_preds[integer_columns]

                    # Find the index of the maximum value in each row using idxmax
                    y_pred_cat = selected_columns.idxmax(axis=1)

                    # You can now assign y_pred_cat to a new column in this_preds if needed
                    this_preds["y_pred_cat"] = y_pred_cat

                    if save_count == 0:
                        if append_to_previous:
                            if os.path.exists(save_to_file):
                                # File exists, read it
                                preds_df = pd.read_csv(save_to_file)
                            else:
                                # File does not exist, create an empty DataFrame with the same columns as this_preds
                                preds_df = pd.DataFrame(columns=this_preds.columns)

                            # Concatenate this_preds to preds_df
                            preds_df = pd.concat([preds_df, this_preds], axis=0)
                        else:
                            preds_df = this_preds
                    else:
                        preds_df = pd.concat([preds_df, this_preds], axis=0)

                    preds_df.to_csv(save_to_file, index=False)
                    print(f"Saved round {save_count}")
                    yhat_for_save = []
                    fns_for_save = []
                    save_count += 1

                ## Save file only if preds_df is not None
                if preds_df is not None:
                    preds_df.to_csv(save_to_file, index=False)
                #                 pdb.set_trace()
                pbar.update()