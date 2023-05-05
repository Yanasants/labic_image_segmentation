from segmentation_models import Unet, Linknet
from unet_original import UnetOriginal
import json

with open("config.json", "r") as params:
    config = json.load(params)

def labic_init(config=config):
    print("Inicializando...")
    backbone = config["backbone_name"]
    if config["model"] == "linknet":
        model = Linknet(backbone_name=config["backbone_name"], encoder_weights=None,
                    input_shape=(None,None,3))
        print(f"Segmentação com modelo Linknet_{backbone} inicializada com sucesso.")
    elif config["model"] == "unet":
        model = Unet(backbone_name=config["backbone_name"], encoder_weights=None,
                    input_shape=(None,None,3))
        print(f"Segmentação com modelo Unet_{backbone} inicializada com sucesso.")
    elif config["model"] == "unet_original":
        unet_original = UnetOriginal(input_layer_shape=(256, 256, 1))
        model = unet_original.generate_model()
        print(f"Segmentação com modelo Unet_original inicializada com sucesso.")
    else:
        raise ValueError("Modelo inválido! Revise o arquivo config.json")