__all__ = [
    "ExperimentRecorder",
    "TrainConfig",
    "build_model",
    "collect_runtime_context",
    "create_dataloaders",
    "train",
]


def __getattr__(name):
    if name == "TrainConfig":
        from .config import TrainConfig

        return TrainConfig
    if name == "create_dataloaders":
        from .data import create_dataloaders

        return create_dataloaders
    if name == "train":
        from .engine import train

        return train
    if name in {"ExperimentRecorder", "collect_runtime_context"}:
        from .experiment import ExperimentRecorder, collect_runtime_context

        return {
            "ExperimentRecorder": ExperimentRecorder,
            "collect_runtime_context": collect_runtime_context,
        }[name]
    if name == "build_model":
        from .model import build_model

        return build_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
