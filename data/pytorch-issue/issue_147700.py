import torch
import torch.nn as nn
import numpy as np

def forward(self, x):
        # Pass inputs through the CNN backbone...
        tokens = self.backbone(x)["layer4"]

        # Pass outputs from the backbone through a simple conv...
        tokens = self.conv1x1(tokens)

        # Re-order in patches format
        tokens = rearrange(tokens, "b c h w -> b (h w) c")

        # Pass encoded patches through encoder...
        out_encoder = self.transformer_encoder((tokens + self.pe_encoder))

        # We expand so each image of each batch get's it's own copy of the
        # query embeddings. So from (1, 100, 256) to (4, 100, 256) for example
        # for batch size=4, with 100 queries of embedding dimension 256.
        queries = self.queries.repeat(out_encoder.shape[0], 1, 1)

        # Compute outcomes for all intermediate
        # decoder's layers...
        class_preds = []
        bbox_preds = []

        for layer in self.transformer_decoder.layers:
            queries = layer(queries, out_encoder)
            class_preds.append(self.linear_class(queries))
            bbox_preds.append(self.linear_bbox(queries))

        # Stack and return
        class_preds = torch.stack(class_preds, dim=1)
        bbox_preds = torch.stack(bbox_preds, dim=1)

        return class_preds, bbox_preds

def run_inference(
    model,
    device,
    inputs,
    nms_threshold=0.3,
    image_size=480,
    empty_class_id=0,
    out_format="xyxy",
    scale_boxes=True,
):
    """
    Utility function that wraps the inference and post-processing and returns the results for the
    batch of inputs. The inference will be run using the passed model and device while post-processing
    will be done on the CPU.

    Args:
        model (torch.nn.Module): The trained model for inference.
        device (torch.device): The device to run inference on.
        inputs (torch.Tensor): Batch of input images.
        nms_threshold (float, optional): NMS threshold for removing overlapping boxes. Default is 0.3.
        image_size (int, optional): Image size for transformations. Default is 480.
        empty_class_id (int, optional): The class ID representing 'no object'. Default is 0.
        out_format (str, optional): Output format for bounding boxes. Default is "xyxy".
        scale_boxes (bool, optional): Whether to scale the bounding boxes. Default is True.
    Returns:
        List of tuples: Each tuple contains (nms_boxes, nms_probs, nms_classes) for a batch item.
    """
    if model and device:
        model.eval()
        model.to(device)
        inputs = inputs.to(device)
    else:
        raise ValueError("No model or device provided for inference!")

    with torch.no_grad():
        out_cl, out_bbox = model(inputs)

    # Get the outputs from the last decoder layer..
    out_cl = out_cl[:, -1, :]
    out_bbox = out_bbox[:, -1, :]
    out_bbox = out_bbox.sigmoid().cpu()
    out_cl_probs = out_cl.cpu()

    scale_factors = torch.tensor([image_size, image_size, image_size, image_size])
    results = []

    for i in range(inputs.shape[0]):
        o_bbox = out_bbox[i]
        o_cl = out_cl_probs[i].softmax(dim=-1)
        o_bbox = ops.box_convert(o_bbox, in_fmt="cxcywh", out_fmt=out_format)

        # Scale boxes if needed...
        if scale_boxes:
            o_bbox = o_bbox * scale_factors

        # Filter out boxes with no object...
        o_keep = o_cl.argmax(dim=-1) != empty_class_id
        if o_keep.sum() == 0:
            results.append((np.array([]), np.array([]), np.array([])))
            continue
        keep_probs = o_cl[o_keep]
        keep_boxes = o_bbox[o_keep]

        # Apply NMS
        nms_boxes, nms_probs, nms_classes = class_based_nms(
            keep_boxes, keep_probs, nms_threshold
        )
        results.append((nms_boxes, nms_probs, nms_classes))

    return results

class DETR(nn.Module):
    """Detection Transformer (DETR) model with a ResNet50 backbone.

    Paper: https://arxiv.org/abs/2005.12872

    Args:
        d_model (int, optional): Embedding dimension. Defaults to 256.
        n_classes (int, optional): Number of classes. Defaults to 92.
        n_tokens (int, optional): Number of tokens. Defaults to 225.
        n_layers (int, optional): Number of layers. Defaults to 6.
        n_heads (int, optional): Number of heads. Defaults to 8.
        n_queries (int, optional): Number of queries/max objects. Defaults to 100.

    Returns:
        DETR: DETR model
    """

    def __init__(
        self,
        d_model=256,
        n_classes=92,
        n_tokens=225,
        n_layers=6,
        n_heads=8,
        n_queries=100,
    ):
        super().__init__()

        self.backbone = create_feature_extractor(
            torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True),
            return_nodes={"layer4": "layer4"},
        )

        self.conv1x1 = nn.Conv2d(2048, d_model, kernel_size=1, stride=1)

        self.pe_encoder = nn.Parameter(
            torch.rand((1, n_tokens, d_model)), requires_grad=True
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.queries = nn.Parameter(
            torch.rand((1, n_queries, d_model)), requires_grad=True
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )

        # Each of the decoder's outputs will be passed through
        # linear layers for prediction of boxes/classes.
        self.linear_class = nn.Linear(d_model, n_classes)
        self.linear_bbox = nn.Linear(d_model, 4)

        # Add hooks to get intermediate outcomes
        self.decoder_outs = {}
        for i, L in enumerate(self.transformer_decoder.layers):
            name = f"layer_{i}"
            L.register_forward_hook(get_hook(self.decoder_outs, name))