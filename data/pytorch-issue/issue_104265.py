import torch
import torch.nn as nn

class ConvNextSmallQAT(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = convnext_small(weights="DEFAULT")
        self.model.classifier[2] = torch.nn.Linear(
            in_features=self.model.classifier[2].in_features, out_features=num_classes
        )
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

class ExperimentExectutor:
    def __init__(
        self,
        experiment_config,
        num_gpus: int = 1,
        num_workers: int = 8,
        convert_model: bool = True,
    ):
        self.batch_size: int = experiment_config.batch_size
        self.max_steps: int = experiment_config.max_steps
        self.optimizer: torch.optim = experiment_config.optimizer
        self.model: torch.nn.Module = experiment_config.model
        self.model_settings: dict = experiment_config.model_settings
        self.data: pl.LightningDataModule = experiment_config.data
        self.num_gpus: int = num_gpus
        self.num_workers: int = num_workers
        self.convert_model: bool = convert_model
        self.experiment_config_name = experiment_config().__class__.__name__

    def __call__(self):
        dm = self.data(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            num_gpus=self.num_gpus,
        )
        if self.prepare_data:
            dm.prepare_data()
            dm.setup()

        model = self.trainer(
            self.model,
            self.model_settings,
            self.label_smoothing,
            self.optimizer_settings,
            self.optimizer,
            self.lr_scheduler_settings,
            self.lr_scheduler,
        )
        compiled_model = torch.compile(model)
        tensorboard_logger = TensorBoardLogger("../outputs/tensorboard/")
        trainer = pl.Trainer(
            max_steps=self.max_steps,
            val_check_interval=self.validation_frequency,
            logger=[ tensorboard_logger],
            accelerator="auto",
            callbacks=[
                checkpoint_callback,
                TQDMProgressBar(refresh_rate=10),
                LearningRateMonitor(logging_interval="step"),
            ],
            devices=self.num_gpus,
            # precision=16,
        )
        print("Before QAT:")
        print_size_of_model(compiled_model)
        compiled_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(compiled_model, inplace=True)

        trainer.fit(compiled_model, dm)