from typing import Optional


try:
    from typing import Literal, Callable, Any
    from .rtpt import RTPT
    import tensorflow as tf
    from tensorflow import keras


except ModuleNotFoundError as err:
    print(
        "\033[31mError: Tensorflow 2.x is not installed. Please run `pip install tensorflow` to install the package and try again.\033[0m"
    )
    exit(0)


class RTPTCallback(keras.callbacks.Callback):
    def __init__(
        self,
        name_initials: str,
        experiment_name: str,
        max_iterations: int,
        subtitle_fn: Optional[Callable[[dict[str, Any]], str]] = None,
    ) -> None:
        super().__init__()

        self.subtitle_fn = subtitle_fn

        self.__rtpt = RTPT(
            name_initials=name_initials,
            experiment_name=experiment_name,
            max_iterations=max_iterations,
        )

    def on_train_begin(self, logs=None):
        print("[RTPT]: Starting Training")
        self.__rtpt.start()

    def on_train_end(self, logs=None):
        print("[RTPT]: Training ended")
        self.__rtpt.step(subtitle="done")

    def on_epoch_end(self, epoch, logs=None):
        subtitle = None if self.subtitle_fn is None else self.subtitle_fn(logs)
        self.__rtpt.step(subtitle=subtitle)
