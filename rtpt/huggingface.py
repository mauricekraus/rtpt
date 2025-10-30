from collections.abc import Sized
from typing import Optional


try:
    from typing import Any, Callable, Literal
    import math
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    from .rtpt import RTPT
except ModuleNotFoundError:
    print(
        "\033[31mError: transformers is not installed. Please run `pip install transformers` to install the package and try again.\033[0m"
    )
    exit(0)


class RTPTCallback(TrainerCallback):
    def __init__(
        self,
        name_initials: str,
        experiment_name: str,
        max_iterations: Optional[int] = None,
        current_job: int = 1,
        num_jobs: int = 1,
        mode: Literal["train", "eval", "both"] = "train",
        subtitle_fn: Optional[Callable[[dict[str, Any]], str]] = None,
    ) -> None:
        super().__init__()

        self.subtitle_fn = subtitle_fn
        self.__mode = mode
        self.num_jobs = num_jobs
        self.current_job = current_job
        self._configured_max_iterations = max_iterations

        self.__name_initials = name_initials
        self.__experiment_name = experiment_name
        self.__rtpt: Optional[RTPT] = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if not state.is_local_process_zero:
            return control

        if self.__rtpt is None:
            train_dataloader = kwargs.get("train_dataloader")
            max_iterations = self.__resolve_max_iterations(args, state, train_dataloader)
            total_iterations = max_iterations * (self.num_jobs - self.current_job + 1)
            print(f"[RTPT]: Found {total_iterations} Iterations")
            self.__rtpt = RTPT(
                name_initials=self.__name_initials,
                experiment_name=f"{self.__experiment_name} ({self.current_job}:{self.num_jobs})",
                max_iterations=total_iterations,
            )

        print(f"[RTPT]: Starting Training {self.current_job}:{self.num_jobs}")
        self.__rtpt.start()
        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if not state.is_local_process_zero or self.__rtpt is None:
            return control

        print("[RTPT]: Training ended")
        self.__rtpt.step(subtitle=f"done ({self.current_job}:{self.num_jobs})")
        return control

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> TrainerControl:
        if not state.is_local_process_zero or logs is None or self.__rtpt is None:
            return control

        if not self.__should_step(logs):
            return control

        subtitle = None if self.subtitle_fn is None else self.subtitle_fn(logs)
        self.__rtpt.step(subtitle=subtitle)
        return control

    def __should_step(self, logs: dict[str, Any]) -> bool:
        if self.__mode == "both":
            return True
        has_eval_metrics = any(key.startswith("eval_") for key in logs.keys())
        if self.__mode == "eval":
            return has_eval_metrics
        return not has_eval_metrics

    def __resolve_max_iterations(
        self,
        args: TrainingArguments,
        state: TrainerState,
        train_dataloader: Optional[Any],
    ) -> int:
        if self._configured_max_iterations is not None:
            return self._configured_max_iterations

        # Prefer the TrainerState.max_steps as it reflects the training loop setup.
        if state.max_steps and state.max_steps > 0:
            return state.max_steps

        # Fall back to the user configured max_steps on TrainingArguments if available.
        if getattr(args, "max_steps", 0) and args.max_steps > 0:
            return args.max_steps

        inferred_from_loader = self.__infer_from_dataloader(args, train_dataloader)
        if inferred_from_loader is not None:
            return inferred_from_loader

        raise ValueError(
            "Unable to infer `max_iterations` automatically. Please provide it explicitly when instantiating RTPTCallback."
        )

    def __infer_from_dataloader(
        self,
        args: TrainingArguments,
        train_dataloader: Optional[Any],
    ) -> Optional[int]:
        if train_dataloader is None or not isinstance(train_dataloader, Sized):
            return None

        try:
            steps_per_epoch = len(train_dataloader)
        except TypeError:
            return None

        if steps_per_epoch == 0:
            return None

        accumulation = max(1, getattr(args, "gradient_accumulation_steps", 1))
        steps_per_epoch = math.ceil(steps_per_epoch / accumulation)

        total_epochs = getattr(args, "num_train_epochs", 0)
        if total_epochs and total_epochs > 0:
            total_steps = math.ceil(steps_per_epoch * total_epochs)
            if total_steps > 0:
                return total_steps

        return None
