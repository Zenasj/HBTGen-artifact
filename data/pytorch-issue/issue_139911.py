import torch.nn as nn

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import fbgemm_gpu.sparse_ops  # noqa: F401

import torch
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.datasets.random import RandomRecDataset
from torchrec.datasets.utils import Batch
from torchrec.distributed.global_settings import set_propogate_device
from torchrec.fx.tracer import Tracer
from torchrec.inference.modules import (
    PredictFactory,
    PredictModule,
    quantize_inference_model,
    shard_quant_model,
)
from torchrec.models.dlrm import DLRM
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

import argparse
import sys
from typing import List

logger: logging.Logger = logging.getLogger(__name__)

def register_fake_classes() -> None:
    @torch._library.register_fake_class("fbgemm::AtomicCounter")
    class FakeAtomicCounter:
        def __init__(self, counter_):
            self.counter_ = counter_
        @classmethod
        def __obj_unflatten__(cls, flat_obj):
            return cls(**dict(flat_obj))
        def increment(self) -> int:
            self.counter_ += 1
            return self.counter_
        def decrement(self) -> int:
            self.counter_ -= 1
            return self.counter_
        def reset(self):
            self.counter_ = 0
        def get(self) -> int:
            return self.counter_
        def set(self, val):
            self.counter_ = val
    @torch._library.register_fake_class("fbgemm::TensorQueue")
    class FakeTensorQueue:
        def __init__(self, queue, init_tensor):
            self.queue = queue
            self.init_tensor = init_tensor
        @classmethod
        def __obj_unflatten__(cls, flattened_ctx):
            return cls(**dict(flattened_ctx))
        def push(self, x):
            self.queue.append(x)
        def pop(self):
            if len(self.queue) == 0:
                return self.init_tensor
            return self.queue.pop(0)
        def top(self):
            if len(self.queue) == 0:
                return self.init_tensor
            return self.queue[0]
        def size(self):
            return len(self.queue)



def create_training_batch(args) -> Batch:
    return RandomRecDataset(
        keys=DEFAULT_CAT_NAMES,
        batch_size=args.batch_size,
        hash_size=args.num_embedding_features,
        ids_per_feature=1,
        num_dense=len(DEFAULT_INT_NAMES),
    ).batch_generator._generate_batch()


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm model packager")
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default="45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,"
        "10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35",
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--sparse_feature_names",
        type=str,
        default=",".join(DEFAULT_CAT_NAMES),
        help="Comma separated names of the sparse features.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--num_dense_features",
        type=int,
        default=len(DEFAULT_INT_NAMES),
        help="Number of dense features.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path of model package.",
    )
    return parser.parse_args(argv)

@dataclass
class DLRMModelConfig:
    """
    Model Config for specifying DLRM model parameters.
    """

    dense_arch_layer_sizes: List[int]
    dense_in_features: int
    embedding_dim: int
    id_list_features_keys: List[str]
    num_embeddings_per_feature: List[int]
    num_embeddings: int
    over_arch_layer_sizes: List[int]
    sample_input: Batch


class DLRMPredictModule(PredictModule):
    """
    nn.Module to wrap DLRM model to use for inference.

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define SparseArch.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (List[int]): the layer sizes for the DenseArch.
        over_arch_layer_sizes (List[int]): the layer sizes for the OverArch. NOTE: The
            output dimension of the InteractionArch should not be manually specified
            here.
        id_list_features_keys (List[str]): the names of the sparse features. Used to
            construct a batch for inference.
        dense_device: (Optional[torch.device]).
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        id_list_features_keys: List[str],
        dense_device: Optional[torch.device] = None,
    ) -> None:
        module = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
            dense_device=dense_device,
        )
        super().__init__(module, dense_device)

        self.id_list_features_keys: List[str] = id_list_features_keys

    def predict_forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch (Dict[str, torch.Tensor]): currently expects input dense features
                to be mapped to the key "float_features" and input sparse features
                to be mapped to the key "id_list_features".

        Returns:
            Dict[str, torch.Tensor]: output of inference.
        """

        try:
            logits = self.predict_module(
                batch["float_features"],
                KeyedJaggedTensor(
                    keys=self.id_list_features_keys,
                    lengths=batch["id_list_features.lengths"],
                    values=batch["id_list_features.values"],
                ),
            )
            predictions = logits.sigmoid()
        except Exception as e:
            logger.info(e)
            raise e

        # Flip predictions tensor to be 1D. TODO: Determine why prediction shape
        # can be 2D at times (likely due to input format?)
        predictions = predictions.reshape(
            [
                predictions.size()[0],
            ]
        )

        return {
            "default": predictions.to(torch.device("cpu"), non_blocking=True).float()
        }


class DLRMPredictFactory(PredictFactory):
    """
    Factory Class for generating TorchScript DLRM Model for C++ inference.

    Args:
        model_config (DLRMModelConfig): model config

    """

    def __init__(self, model_config: DLRMModelConfig) -> None:
        self.model_config = model_config

    def create_predict_module(self, world_size: int, device: str) -> torch.nn.Module:
        logging.basicConfig(level=logging.INFO)
        set_propogate_device(True)

        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=self.model_config.embedding_dim,
                num_embeddings=(
                    self.model_config.num_embeddings_per_feature[feature_idx]
                    if self.model_config.num_embeddings is None
                    else self.model_config.num_embeddings
                ),
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(
                self.model_config.id_list_features_keys
            )
        ]
        ebc = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))

        module = DLRMPredictModule(
            embedding_bag_collection=ebc,
            dense_in_features=self.model_config.dense_in_features,
            dense_arch_layer_sizes=self.model_config.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.model_config.over_arch_layer_sizes,
            id_list_features_keys=self.model_config.id_list_features_keys,
            dense_device=device,
        )

        quant_model = quantize_inference_model(module)
        sharded_model, _ = shard_quant_model(
            quant_model, compute_device=device, sharding_device=device
        )



        batch = {}
        batch["float_features"] = self.model_config.sample_input.dense_features.to(
            device
        )
        batch["id_list_features.lengths"] = (
            self.model_config.sample_input.sparse_features.lengths().to(device)
        )
        batch["id_list_features.values"] = (
            self.model_config.sample_input.sparse_features.values().to(device)
        )



        sharded_model(batch)

        aot_compile_options = {
            "aot_inductor.output_path": os.path.join(os.getcwd(), "dlrm_pt2.so"),
        }



        #with torch.no_grad():
        with torch.inference_mode():
            exported_program = torch.export.export(
                sharded_model,
                (batch,),
                strict=False,
            )
        so_path = torch._inductor.aot_compile(
        exported_program.module(),
        (batch,),
        # Specify the generated shared library path
        options=aot_compile_options
    )

    def batching_metadata(self) -> Dict[str, str]:
        return {
            "float_features": "dense",
            "id_list_features": "sparse",
        }

    def result_metadata(self) -> str:
        return "dict_of_tensor"

    def run_weights_independent_tranformations(
        self, predict_module: torch.nn.Module
    ) -> torch.nn.Module:
        return predict_module

    def run_weights_dependent_transformations(
        self, predict_module: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Run transformations that depends on weights of the predict module. e.g. lowering to a backend.
        """
        return predict_module


def main(argv: List[str]) -> None:
    """
    Use torch.package to package the torchrec DLRM Model.

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """

    args = parse_args(argv)

    args.batch_size = 10
    args.num_embedding_features = 26
    batch = create_training_batch(args)

    register_fake_classes()

    model_config = DLRMModelConfig(
        dense_arch_layer_sizes=list(map(int, args.dense_arch_layer_sizes.split(","))),
        dense_in_features=args.num_dense_features,
        embedding_dim=args.embedding_dim,
        id_list_features_keys=args.sparse_feature_names.split(","),
        num_embeddings_per_feature=list(
            map(int, args.num_embeddings_per_feature.split(","))
        ),
        num_embeddings=args.num_embeddings,
        over_arch_layer_sizes=list(map(int, args.over_arch_layer_sizes.split(","))),
        sample_input=batch,
    )

    DLRMPredictFactory(model_config).create_predict_module(world_size=1, device="cuda")



if __name__ == "__main__":
    main(sys.argv[1:])

def forward(self, int_nbit_split_embedding_codegen_lookup_function):
        _module_sparse_arch_embedding_bag_collection_tbes_0_lxu_cache_locations_list = getattr(self._module.sparse_arch.embedding_bag_collection.tbes, "0").lxu_cache_locations_list
        call_torchbind = torch.ops.higher_order.call_torchbind(_module_sparse_arch_embedding_bag_collection_tbes_0_lxu_cache_locations_list, 'pop');  _module_sparse_arch_embedding_bag_collection_tbes_0_lxu_cache_locations_list = call_torchbind = None
        return (int_nbit_split_embedding_codegen_lookup_function,)