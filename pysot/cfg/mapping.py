from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

MODEL_CONFIG_WEIGHTS_MAP = {
    "siammask_r50_l3": {
        "config": "config.yaml",
        "weights": "https://drive.google.com/file/d/1dQoI2o5Bzfn_IhNJNgcX4OE79BIHwr8s/view?usp=drive_link",
        "description": "SiamMask model with a ResNet-50 backbone and 3-layer feature extraction for object tracking and segmentation.",
    },
    "siamrpn_alex_dwxcorr": {
        "config": "config.yaml",
        "weights": "https://drive.google.com/file/d/1e51IL1UZ-5seum2yUYpf98l2lJGUTnhs/view?usp=drive_link",
        "description": "SiamRPN model with AlexNet backbone and depthwise cross-correlation for efficient object tracking.",
    },
    "siamrpn_alex_dwxcorr_16gpu": {
        "config": "config.yaml",
        "weights": "https://dl.fbaipublicfiles.com/siammask/siammask_r50_l3.pth",
        "description": "SiamRPN model with AlexNet backbone and depthwise cross-correlation, optimized for 16-GPU training setups.",
    },
    "siamrpn_alex_dwxcorr_otb": {
        "config": "config.yaml",
        "weights": "https://drive.google.com/file/d/1cZFJgPfLZodtBpxA7XJPJ2sErJ56KSVk/view?usp=drive_link",
        "description": "SiamRPN model with AlexNet backbone and depthwise cross-correlation, fine-tuned for the OTB benchmark dataset.",
    },
    "siamrpn_mobilev2_l234_dwxcorr": {
        "model": "siamrpn_mobilev2_l234_dwxcorr",
        "config": "config.yaml",
        "weights": "https://drive.google.com/file/d/1lPiRjFvajwrhOHVuXrygAj2cRb2BFFfz/view?usp=drive_link",
        "description": "SiamRPN model with MobileNetV2 backbone and layers 2, 3, and 4 for lightweight and efficient tracking.",
    },
    "siamrpn_r50_l234_dwxcorr_lt": {
        "config": "config.yaml",
        "weights": "https://drive.google.com/file/d/17rY2dJU1UF0BozPI0zGvw2c5sWrSQuwt/view?usp=drive_link",
        "description": "SiamRPN model with ResNet-50 backbone and layers 2, 3, and 4, fine-tuned for long-term tracking scenarios.",
    },
    "siamrpn_r50_l234_dwxcorr_otb": {
        "config": "config.yaml",
        "weights": "https://drive.google.com/file/d/17sRbpvAzcHAu5bligZTD9QM8XHoJVDEY/view?usp=drive_link",
        "description": "SiamRPN model with ResNet-50 backbone and layers 2, 3, and 4, fine-tuned for the OTB benchmark dataset.",
    },
}


class ModelConfigBuilder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_config = MODEL_CONFIG_WEIGHTS_MAP.get(model_name)
        if not self.model_config:
            raise ValueError(
                f"Model '{model_name}' not found in the configuration mapping."
            )

    def get_model(self) -> dict:
        return self.model_config

    def get_weights_url(self) -> str:
        return self.model_config["weights"]

    def get_description(self) -> str:
        return self.model_config["description"]

    def get_metadata(self) -> dict:
        return {
            "name": self.model_name,
            "dir": ROOT_DIR / "cfg" / "models" / self.model_name,
            "config": ROOT_DIR
            / "cfg"
            / "models"
            / self.model_name
            / self.model_config["config"],
            "weights": self.get_weights_url(),
            "description": self.get_description(),
        }
