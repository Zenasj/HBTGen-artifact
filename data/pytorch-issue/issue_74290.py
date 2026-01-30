import torch

import pathlib

from torch import package
import transformers

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    num_labels=5,
)

_RESOURCE_NAME = "models"
_PACKAGE_NAME = "frozen_model"
_FILE_NAME = "frozen_model.pt"


def freeze_model(model, save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = (save_path / _FILE_NAME)
    with package.PackageExporter(file_path) as exp:
        exp.extern("transformers.**")
        exp.extern("torch.**")
        exp.save_pickle(_PACKAGE_NAME, _RESOURCE_NAME, model)


def load_model(path):
    importer = package.PackageImporter(path)
    runner = importer.load_pickle(_PACKAGE_NAME, _RESOURCE_NAME)
    return runner


freeze_model(model, pathlib.Path("./out"))
load_model(pathlib.Path("./out/frozen_model.pt"))