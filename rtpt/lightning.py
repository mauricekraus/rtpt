from typing import Optional


try:
    from lightning.pytorch.callbacks import Callback
    from typing import Literal, Callable, Any
    from .rtpt import RTPT
    from lightning import Trainer, LightningModule


except ModuleNotFoundError as err:
    print(
        "\033[31mError: pytorch-lightning is not installed. Please run `pip install pytorch-lightning` to install the package and try again.\033[0m"
    )
    exit(0)


class RTPTCallback(Callback):
    def __init__(
        self,
        name_initials: str,
        experiment_name: str,
        max_iterations: int,
        mode: Literal["val", "train"] = "train",
        subtitle_fn: Optional[Callable[[dict[str, Any]], str]] = None,
    ) -> None:
        super().__init__()

        self.subtitle_fn = subtitle_fn

        self.__rtpt = RTPT(
            name_initials=name_initials,
            experiment_name=experiment_name,
            max_iterations=max_iterations,
        )
        self.__mode = mode

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        print("[RTPT]: Starting Training")
        self.__rtpt.start()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        print("[RTPT]: Training ended")
        self.__rtpt.step(subtitle="done")

    def on_step(self, trainer: Trainer):
        subtitle = (
            None
            if self.subtitle_fn is None
            else self.subtitle_fn(trainer.callback_metrics)
        )
        self.__rtpt.step(subtitle=subtitle)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.__mode == "train":
            self.on_step(trainer)

    def on_val_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.__mode == "val":
            self.on_step(trainer)
