import numpy as np
import onnxruntime as rt
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import convert_graph_to_onnx


class LabseForClassification(nn.Module):
    def __init__(self, config):
        super(LabseForClassification, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.labse = AutoModel.from_config(config)
        self.classifier = nn.Linear(768, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        model_output = self.labse(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        embeddings = model_output[1]
        embeddings = nn.functional.normalize(embeddings)
        logits = self.classifier(embeddings)
        return torch.softmax(logits, dim=-1)


TEXT = """LOS ANGELES (KTLA) – In high school football, the talent gap between future pros and average Joes can be quite wide indeed, and lopsided scores are nothing new. But a 106-0 win? That will draw some attention. That was the score at the game between Inglewood High and Inglewood Morningside in Southern California over the weekend. Despite scoring 59 points in the first quarter alone, Inglewood High head coach Mil’Von James declined to play backups and was initially reticent to use a running clock to shorten the game, according to Inglewood Morningside football coach Brian Collins. Inglewood High even went for a two-point conversion pass, instead of the traditional one-point kick attempt, after scoring to take a triple-digit lead, which Collins told the Los Angeles Times was “a classless move.” “I told them, ‘Go play St. John Bosco and Mater Dei,'” Collins said in reference to two of the area’s powerhouse high schools that recently produced the starting quarterbacks at top-tier programs Clemson University and the University of Alabama. James has not responded to an email seeking comment on the game. In a statement provided to the Times’ Eric Sondheimer, the California Interscholastic Federation Southern Section, which governs most Southern California high school sports, said the 106-0 score “does not represent” the organization’s ideals of character. “The CIF-SS condemns, in the strongest terms, results such as these,” the statement read. Other high school coaches were similarly incensed. Matt Poston, head coach at Tesoro High School in Las Flores, said he hoped he was “reading this wrong” when he looked at the score. “We’re supposed to be teaching young men life lessons through the game. What message was this staff teaching last night? Sad,” Poston wrote on Twitter. Legendary basketball sportscaster Dick Vitale also weighed in on Twitter. Sportswriter Nick Harris highlighted some of the most eye-popping stats, calling the game “a beatdown for the ages.” While 106-0 is a score rarely seen at any level of football, it’s not the largest margin of victory. The most lopsided football score of all time is widely considered to be Georgia Tech’s 222-0 win over Cumberland in 1916, when Cumberland had discontinued its football program but was forced to play the game, putting together a squad of fraternity brothers and other students."""
ENTITY = "Inglewood"
TEXT = TEXT.replace(ENTITY, "[MASK]")

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE", use_fast=True)
    config = AutoConfig.from_pretrained("sentence-transformers/LaBSE", num_labels=2)

    model_raw = LabseForClassification(config)
    model_raw.load_state_dict(torch.load("outputs/SAL/pytorch_model.bin", map_location="cpu"))
    model_raw.eval()

    model_pipeline = transformers.Pipeline(model=model_raw, tokenizer=tokenizer)

    with torch.no_grad():
        input_names, output_names, dynamic_axes, tokens = convert_graph_to_onnx.infer_shapes(
            model_pipeline,
            "pt"
        )
        ordered_input_names, model_args = convert_graph_to_onnx.ensure_valid_input(
            model_pipeline.model, tokens, input_names
        )

    del dynamic_axes["output_0"]  # Delete unused output

    output_names = ["probs"]
    dynamic_axes["probs"] = {0: 'batch'}

    torch.onnx.export(
        model_raw,
        model_args,
        f="test.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=12,
    )

    sess = rt.InferenceSession("test.onnx")
    inputs_np = tokenizer(TEXT, return_tensors="np")
    probs_onnx = sess.run(None, {
        "input_ids": inputs_np["input_ids"],
        "attention_mask": inputs_np["attention_mask"],
        "token_type_ids": inputs_np["token_type_ids"]
    })

    inputs = tokenizer(TEXT, return_tensors="pt")
    probs = model_raw(**inputs)

    assert np.allclose(
        probs_onnx[0].squeeze(),
        probs.squeeze().detach().numpy(),
        atol=1e-6,
    )