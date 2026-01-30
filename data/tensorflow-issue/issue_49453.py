from tensorflow.keras import layers

training = model.fit(
        generator=train_data_generator,
        epochs=EPOCHS_NUMBER,
        validation_data=valid_data_generator,
        callbacks=callbacks,
    )

training = model.fit_generator(
        generator=train_data_generator,
        epochs=EPOCHS_NUMBER,
        validation_data=valid_data_generator,
        callbacks=callbacks,
    )

from pathlib import Path

import mlflow.tensorflow
from sklearn.model_selection import GroupKFold, StratifiedKFold
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from classes.Evaluator import Evaluator
from utils.args import get_args_train
from utils.inputs import get_inputs_paths_and_targets
from utils.model import build_model, get_preprocess_function
from utils.seed import set_seeds

# Setting the random_state to get reproducible results
seed = set_seeds()

# Constants
ROOT_FOLDER = Path(__file__).resolve().parent.parent
FOLDS_NUMBER = 5
EPOCHS_NUMBER = 100
EPOCHS_WAITING_FOR_IMPROVEMENT = 5

# Gets which input we're going to use
args = get_args_train()
IMAGE_FOLDER_PATH = ROOT_FOLDER / f"crop_result_prop_{args.proportion}"


class TrainingAndValidationMetrics(Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics(logs, step=epoch)


# Instantiates GroupKFold class to split into train and test
group_k_fold = GroupKFold(n_splits=FOLDS_NUMBER)

# Configuration dictionary that is going to be used to compile the model
config = {
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "metrics": ["accuracy", Precision(), Recall()],
}

# Gets the dataframe that are going to be used in the flow_from_dataframe
# method from the ImageDataGenerator class
input_df = get_inputs_paths_and_targets(args.proportion)

idg = ImageDataGenerator(
    fill_mode="nearest",
    preprocessing_function=get_preprocess_function(args.model.lower()),
)

# Using GroupKFold only to guarantee that a group (in this case, the slide)
# will contain data only in the train or the test group
for train_idx, test_idx in group_k_fold.split(
    input_df.input, input_df.target, input_df.slide
):
    train_data = input_df.iloc[train_idx]
    test_data = input_df.iloc[test_idx]

    # Break here is being used to get only the first fold
    break

print("### Testing data distribution ###")
print(f"{test_data.groupby('slide').count()}")

generator_kwargs = {
    "directory": IMAGE_FOLDER_PATH,
    "x_col": "input",
    "y_col": "target",
    "seed": seed,
    "classes": ["0", "1"],
}

# Generator that will be used to evaluate the model
test_data_generator = idg.flow_from_dataframe(
    test_data, class_mode="binary", shuffle=False, **generator_kwargs
)

# Instantiates the Early Stopping that is going to be used
# in the fit method
early_stopping = EarlyStopping(
    monitor="val_loss", patience=EPOCHS_WAITING_FOR_IMPROVEMENT
)

# Defines the list of callbacks that is an argument of the fit
# method
callbacks = [early_stopping, TrainingAndValidationMetrics()]

# K-fold Cross Validation model evaluation
kfold = StratifiedKFold(n_splits=FOLDS_NUMBER, shuffle=True, random_state=seed)

current_fold = 1
# Starts run on mlflow to register metrics (experiment)
mlflow.start_run(
    run_name=f"{args.model.lower()}",
    tags={"data_proportion": args.proportion, "environment": "main"},
)


# Folds iteration
for train_idx, val_idx in kfold.split(train_data.input, train_data.target):

    # Builds the model
    model = build_model(args.model.lower(), [Dense(1, activation="sigmoid")], config)

    # Starts run on mlflow to register metrics (runs)
    with mlflow.start_run(run_name=f"fold_{current_fold}", nested=True):

        fitting_data = train_data.iloc[train_idx]
        val_data = train_data.iloc[val_idx]

        mlflow.log_text(
            f"Training data: \n{fitting_data.groupby('target').count()} \n"
            f"Validation data: \n{val_data.groupby('target').count()} \n"
            f"Data proportion: {args.proportion} \n"
            f" ----- Using Stratified KFold with seed 1 ----- ",
            artifact_file="data_description.txt",
        )

        train_data_generator = idg.flow_from_dataframe(
            fitting_data, class_mode="binary", **generator_kwargs
        )
        valid_data_generator = idg.flow_from_dataframe(
            val_data, class_mode="binary", **generator_kwargs
        )

        training = model.fit(
            train_data_generator,
            epochs=EPOCHS_NUMBER,
            validation_data=valid_data_generator,
            callbacks=callbacks,
        )

        # Logging model
        mlflow.keras.log_model(keras_model=model, artifact_path="model")

        evaluator = Evaluator(model, training, test_data_generator, test_data.target)

        # Classification report
        test_metrics = evaluator.evaluate_model()
        mlflow.log_metrics(test_metrics)

        # Saves the classification report generated from the predict method
        mlflow.log_text(
            evaluator.generate_classification_report(),
            artifact_file="classification_report.txt",
        )

        # Saves the accuracy and the loss per epoch
        mlflow.log_figure(
            evaluator.generate_training_history_image(),
            artifact_file="accuracy_loss_epochs.png",
        )

        # Saves ROC figure
        mlflow.log_figure(evaluator.generate_roc_figure(), artifact_file="roc.png")

    current_fold += 1

mlflow.end_run()