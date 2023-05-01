from segmentation_models import Unet, Linknet
def labic_init():
    print("Inicializando...")
    model = Linknet(backbone_name='resnet34', encoder_weights=None,
                  input_shape=(None,None,3))