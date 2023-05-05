# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 19:04:14 2023

@author: Labic
"""
from skimage import io
import cv2
import os


from skimage.util import img_as_float
from tensorflow import keras

import tensorflow as tf

#from keras.callbacks.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import glob
import time
import random
import datetime
import pandas as pd
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split

from segmentation_models import Unet, Linknet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

# Data augmentation 1 - 2022.05.07 Acrescentando DA no conj de valid
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.data import AUTOTUNE
from tensorflow.keras.optimizers import Adam

class Dataset:
    def __init__(self, folder:str, norm_imgs_folder:str, gt_folder:str, ORIGINAL_SIZE=None, NEW_SIZE=None):
        '''
        Construtor da classe Dataset.
        Após a inicialização, teremos:
            - No atributo X: todas as imagens;
            - No atributo Y: todas as máscaras;

        :param folder: Diretório onde se encontra os subdiretórios com imagens e com as máscaras.
        :type folder: str
        :param norm_imgs_folder: Subdiretório com as imagens.
        :type norm_imgs_folder: str
        :param gt_folder: Subdiretório com as máscaras.
        :type gt_folder: str
        :param ORIGINAL_SIZE: Size original das imagens.
        :type ORIGINAL_SIZE: int
        :param NEW_SIZE: Novo size das imagens.
        :type NEW_SIZE: int

        '''
        self.folder = folder
        self.norm_imgs_folder = norm_imgs_folder
        self.gt_folder = gt_folder
        self.ORIGINAL_SIZE = ORIGINAL_SIZE
        self.NEW_SIZE = NEW_SIZE
        self.X = None
        self.Y = None
        self.X_train, self.X_val, self.Y_train, self.Y_val = (None, None, None, None)
        self.img_shape = None
        self.norm_imgs, self.GT_imgs = None, None
        self.load_images()

    def resize_one_img(self, img: np.ndarray, width:int, height:int):
        '''
        Redimensiona uma imagem.

        :param img: Imagem original.
        :type img: numpy.ndarray
        :param width: Nova largura da imagem.
        :type width: int
        :param height: Nova altura da imagem.
        :type width: int
        :return: Imagem redimensionada.
        :rtype: numpy.ndarray

        '''
        self.curr_img = cv2.resize(img, (width, height))
        return self.curr_img
        
    def load_images_array(self, img_list:list, original_size=160, new_size = None):

        '''
        Recebe um glob das imagens e converte em um numpy array no formato que o Keras aceita.

        :param img_list: Lista com todos os nomes das imagens no diretório. 
        :type img_list: list
        :param new_size: Novo size da imagem (equivalendo para largura e altura).
        :type new_size: int

        :return: Conjunto de imagens no formato de input do Keras [(exemplo formato Keras: (5, 256, 256, 1)] 
        e uma tupla com a altura e a largura, respectivamente, da imagem original.
        :rtype: tuple

        '''
        img = np.zeros((len(img_list), new_size, new_size), dtype=float)
        img_shape = img_as_float(io.imread(img_list[0])).shape
        for i in range(len(img_list)):
            
            im = np.copy(img_as_float(io.imread(img_list[i])))
            im = self.resize_one_img(im, new_size, new_size)
            img[i] = im
    
        # Padrão Keras
        img = img.reshape(-1, img.shape[-2], img.shape[-1], 1)
        return img, img_shape
        
    def load_images(self):

        '''
        Organiza a lista das imagens no diretório e chama a função load_images_array para alimentar 
        os atributos X, Y e img_shape da classe.        
        '''

        self.norm_imgs = sorted(glob.glob(f"{self.folder}{self.norm_imgs_folder}")) 
        self.GT_imgs = sorted(glob.glob(f"{self.folder}{self.gt_folder}")) 
                                        
        for i in range(len(self.norm_imgs)):
            if self.norm_imgs[i][-8:-4] != self.GT_imgs[i][-8:-4]:
                print('Algo está errado com as imagens')

        self.X, self.img_shape = self.load_images_array(img_list=self.norm_imgs, original_size=self.ORIGINAL_SIZE, new_size = self.NEW_SIZE)
        self.Y, self.img_shape = self.load_images_array(img_list=self.GT_imgs, original_size=self.ORIGINAL_SIZE, new_size = self.NEW_SIZE)
        print("\nImagens carregadas com sucesso.")
    
    def split_dataset(self, seed_min=0, seed_max =2**20, test_size=0.2):

        '''
        Separa as imagens e as máscaras de treino e validação.
        Alimenta os atributos X_train, Y_train, X_val e Y_val.

        :param seed_min: Valor mínimo para a semente do random. 
        :type seed_min: int
        :param seed_max: Novo size da imagem (equivalendo para largura e altura).
        :type seed_max: int
        :param test_size: Tamanho do conjunto de teste [Valores entre 0 e 1. Ex.: 0.2 = 20% do total dos dados para teste]. 
        :type test_size: float

        '''

        random.seed(time.time())
        SEED_1 = random.randint(seed_min, seed_max)

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X, self.Y, test_size=test_size, random_state=SEED_1)
        print(f"\nDataset subdividido.\n{test_size*100}% dos dados para validação.\n{(1-test_size)*100}% dos dados para treino.")
                
        
    
class DataAugmentation:
    def __init__(self, X_train:np.ndarray, Y_train:np.ndarray, use_batch_size:int, X_val:np.ndarray, \
                 Y_val:np.ndarray, factor=0.2, direction="horizontal", rotation=0.1):
        '''
        Construtor da classe DataAugmentation.
        Após a inicialização, teremos:
            - No atributo trainDS: dados de treino após o processo de data augmentation;
            - No atributo valDS: dados de validação após o processo de data augmentation;;

        :param X_train: Imagens de treino.
        :type X_train: numpy.ndarray
        :param Y_train: Máscaras de treino.
        :type Y_train: numpy.ndarray

        :param X_val: Imagens de validação.
        :type X_val: numpy.ndarray
        :param Y_val: Máscaras de validação.
        :type Y_val: numpy.ndarray

        :param use_batch_size: Tamanho do pacote (batch-size).
        :type use_batch_size: int
        :param factor: Fator para ajuste no data augmentation. 
        :type factor: float
        :param direction: Direção ("horizontal", "vertical") do ajuste no data augmentation.
        :type direction: str
        :param rotation: Ângulo de rotação das imagens no data augmentation.
        :type rotation: int

        '''
        self.trainAug = None
        self.valAug = None

        self.factor = factor 
        self.direction = direction
        self.rotation = rotation

        self.X_train, self.Y_train, self.X_val, self.Y_val = X_train, Y_train, X_val, Y_val
        self.use_batch_size = use_batch_size

        self.trainDS = None
        self.valDS = None
        self.trainDS, self.valDS = self.augmentation()
        
    @tf.autograph.experimental.do_not_convert # Evita warnings
    def augmentation(self):

        '''
        Aplica o processo de data augmentation de acordo com os parâmetros repassados.
        '''

        self.trainAug = Sequential([
            preprocessing.RandomFlip(self.direction),
            preprocessing.RandomZoom(
                height_factor=(-self.factor, +self.factor),
                width_factor=(-self.factor, +self.factor)),
            preprocessing.RandomRotation(self.rotation)
        ])

        self.valAug = Sequential([
            preprocessing.RandomFlip(self.direction),
            preprocessing.RandomZoom(
                height_factor=(-self.factor, +self.factor),
                width_factor=(-self.factor, +self.factor)),
            preprocessing.RandomRotation(self.rotation)
        ])

        self.data_gen_args = dict(shear_range=self.factor,
                            zoom_range=self.factor,
                            horizontal_flip=True,
                            validation_split=0.1)

        self.image_datagen = ImageDataGenerator(**self.data_gen_args)

        random.seed(time.time())
        seed_min = 0
        seed_max = 2**20
        SEED_2 = random.randint(seed_min, seed_max)

        # Data Augmentation 2 - 2022.05.07 Fazendo DA no conj de valid
        self.trainDS = tf.data.Dataset.from_tensor_slices((self.X_train, self.Y_train))
        self.trainDS = self.trainDS.repeat(3)
        self.trainDS = (
            self.trainDS
            .shuffle(self.use_batch_size * 100)
            .batch(self.use_batch_size)
            .map(lambda x, y: (self.trainAug(x), self.trainAug(y)), num_parallel_calls=AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        self.valDS = tf.data.Dataset.from_tensor_slices((self.X_val, self.Y_val))
        self.valDS = self.valDS.repeat(3)
        self.valDS = (
            self.valDS
            .shuffle(self.use_batch_size * 100)
            .batch(self.use_batch_size)
            .map(lambda x, y: (self.valAug(x), self.valAug(y)), num_parallel_calls=AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
        return self.trainDS, self.valDS



class SegmentationModel:
    def __init__(self, N:int, segmentation_model:str, backbone_name:str, trainDS, valDS, epochs:int, callback=None):
        # falta adicionar o segmentation_model como parâmetro
        '''
        Construtor da classe SegmentationModel.
        Após a inicialização, teremos:
            - No atributo model: Modelo da rede;
            - No atributo history: Report do treinamento;

        :param N: Número de canais (precisa ser 3 ser enconder_weights != None).
        :type N: int
        :param backbone_name: Backbone integrado à rede.
        :type backbone_name: str
        :param trainDS: Conjunto de dados de treino após o data augmentation.
        :type trainDS: tensorflow.python.data.ops.dataset_ops.PrefetchDataset
        :param val_DS: Conjunto de dados de validação após o data augmentation.
        :type val_DS: tensorflow.python.data.ops.dataset_ops.PrefetchDataset
        :param epochs: Número de épocas.
        :type epochs: int
        :param callback: Callback do TensorFlow.
        :type callback: tensorflow.keras.callbacks

        '''
        self.N = N
        self.backbone_name = backbone_name
        self.trainDS, self.valDS = trainDS, valDS
        self.epochs = epochs
        self.callback = callback
        self.model = None
        self.history = None
        self.segmentation_model = segmentation_model
        self.model, self.history = self.generate_model()

    def generate_model(self):

        '''
        Gera o modelo.
        Alimenta os atributos model e history.

        :return: Modelo e History
        :rtype: keras.engine.functional.Functional, keras.callbacks.History
        '''
        if self.segmentation_model=='linknet':
            model = Linknet(backbone_name=self.backbone_name, encoder_weights=None,
                        input_shape=(None,None,self.N))
        if self.segmentation_model=='unet':
            model = Unet(backbone_name=self.backbone_name, encoder_weights=None,
                    input_shape=(None,None,self.N))


        model.compile(optimizer=Adam(), loss=bce_jaccard_loss, metrics=[iou_score]) 

        if self.callback == None:
            history = model.fit(self.trainDS, 
                    epochs=self.epochs, 
                    validation_data=self.valDS)
        else:
            history = model.fit(self.trainDS, 
                epochs=self.epochs, callbacks=self.callback,
                validation_data=self.valDS)
            
        return model, history




    
class SaveReport:
    def __init__(self, model, history, folder_name:str, n_fold:int,epochs:int, exec_folder_name:str, use_batch_size=4):
        '''
        Construtor da classe SaveReport.
        Salva o modelo.

        :param model: Modelo.
        :type model: keras.engine.functional.Functional
        :param history: History com report do treinamento.
        :type history: keras.callbacks.History
        :param folder_name: Diretório onde será armazenado o output.
        :type folder_name: str
        :param n_fold: Número da execução.
        :type n_fold: int
        :param epochs: Número de épocas.
        :type epochs: int
        :param exec_folder_name: Diretório do conjunto atual de execuções. [Ex.: Exec_2023-05-04-20-30-56.952912]
        :type exec_folder_name: str
        :param use_batch_size: Tamanho do pacote (batch-size).
        :type use_batch_size: int

        '''
        self.folder_name = folder_name
        self.n_fold = n_fold
        self.epochs = epochs
        self.use_batch_size = use_batch_size
        self.model = model
        self.history = history

        self.dir_predict = None
        self.exec_folder_name = exec_folder_name
        self.save_model()
        self.save_history()

        
    def create_folder(self, dirName):
        '''
        Cria o diretório se não existir.

        :param dirName: Nome do diretório a ser criado.
        :type dirName: str
        '''
        # Create target Directory if don't exist
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Diretório " , dirName ,  " criado.")
        else:    
            print("Diretório " , dirName ,  " já existe.")
    
    def organize_folders(self):
        '''
        Organiza os diretórios.
        Dentro do output_folder haverá (ou será criada) a pasta outputs.
        Dentro da pasta outputs, por sua vez, estarão os diretórios do conjunto de execuções (por padrão, iniciados com Exec_).
        Dentro de cada pasta de execução, estarão os diretórios de cada n execução.
        [Ex. Path: output_folder/outputs/Exec_folder/fold_n]

        :return: Momento da execução, path da pasta da execução de número n, path da pasta do conjunto de execuções.
        :rtype: str, str, str
        '''
        if (self.n_fold == 0):
            exec_moment = str(datetime.datetime.now()).replace(':','-').replace(' ','-') 
            self.output_folder = f"/outputs/Exec_{exec_moment}"
            self.exec_folder_name = f"{self.folder_name}{self.output_folder}"
        else:
            self.output_folder= f"/outputs/{self.exec_folder_name}"
            self.exec_folder_name = f"{self.folder_name}{self.output_folder}"
            exec_moment = self.exec_folder_name.split('/')[-1].split('_')[1]
            
        self.create_folder(self.exec_folder_name)
        self.dir_predict = f"{self.output_folder}/fold_{self.n_fold}"
        n_fold_folder_name = f"{self.exec_folder_name}/fold_{self.n_fold}"
        self.create_folder(n_fold_folder_name)
        return exec_moment, n_fold_folder_name, self.exec_folder_name
    
    def save_model(self):
        '''
        Alimenta o atributo n_fold_folder_name, organiza as pastas chamando a função organize_folders.
        Salva o modelo no arquivo .h5.
        '''
        exec_moment, self.n_fold_folder_name, exec_folder_name = self.organize_folders()
        self.name_file = "model_"+ str(self.use_batch_size) + "_" + str(self.epochs) + "_exec_%s"%(exec_moment) + "_fold_%i"%self.n_fold
        self.model_name = self.n_fold_folder_name + '/%s.h5'%self.name_file
        self.model.save(self.model_name)
        print(f"\nModelo salvo.\nNome: {self.name_file}\nSalvo em: {exec_folder_name}")

    def save_history(self):
        '''
        Salva o history do treinamento em formato .csv e .txt
        O arquivo .txt necessário pois, apesar do formato .csv facilitar o carregamento de dados a posteriori,
        o armazenamento neste pode ter falhas de caracteres, sobretudo em valores muito altos na Loss.
        '''
        with open(f"{self.n_fold_folder_name}/history_{self.n_fold}.txt", "w") as file:
            file.write(str(self.history.history))

        df = pd.DataFrame(self.history.history)
        df.to_csv(f"{self.n_fold_folder_name}/history_{self.n_fold}.csv")
        print(f"Arquivos history_{self.n_fold}.csv/txt salvos com sucesso.")
        

class PredictImages:
    def __init__(self, test_images, n_fold_folder_name:str, model_name:str, use_batch_size:int, img_shape:tuple):
        '''
        Construtor da classe PredictImages.
        Chama a função predict para predição das imagens de teste.

        :param test_images: Imagens de teste.
        :type test_images: labic_images_segmentation.Dataset
        :param n_fold_folder_name: Path para o diretório da execução n. [Ex.: ./TM40_46Prod/outputs/Exec_2023-05-04-20-30-56.952912/fold_0]
        :type n_fold_folder_name: str
        :param model_name: Path para o arquivo com o modelo. [Ex.: ./TM40_46Prod/outputs/Exec_2023-05-04-20-30-56.952912/fold_0/model_4_3_exec_2023-05-04-20-30-56.952912_fold_0.h5]
        :type model_name: str
        :param use_batch_size: Tamanho do pacote (batch-size).
        :type use_batch_size: int
        :param img_shape: Altura e largura das imagens originais.
        :type img_shape: tuple

        '''
        self.test_images = test_images
        self.model_name = model_name
        self.n_fold_folder_name = n_fold_folder_name
        self.dir_predict = n_fold_folder_name
        self.batch = use_batch_size
        self.img_shape = img_shape

        self.new_predicao = None

        self.predict()

    def predict(self):
        '''
        Carrega o modelo, cria o diretório outputs_prod dentro da pasta da execução n. 
        Realiza e salva as predições, redimensionando as images no momento de salvar.
        '''
        model = keras.models.load_model(self.model_name, compile=False)
        self.new_predicao = model.predict(self.test_images.X)
        self.new_predicao = np.uint8(255*(self.new_predicao > 0.5))

        SaveReport.create_folder(self, self.n_fold_folder_name + '/outputs_prod')
        for i in range(len(self.new_predicao)):
            io.imsave(self.n_fold_folder_name + '/outputs_prod/predicao_%s_%s.png'%(str(self.test_images.GT_imgs[i][-7:-4]), str(self.batch)),\
                       Dataset.resize_one_img(self, self.new_predicao[i], self.test_images.img_shape[1], self.test_images.img_shape[0]))
        
        print("\nImagens preditas com sucesso.")


class DiceCoef(Dataset):
    def __init__(self, gt_imgs:np.ndarray, pred_folder:str, new_size:int):
        '''
        Construtor da classe DiceCoef.
        Após a inicialização, teremos:
            - No atributo pred_imgs: Imagens de teste (em formato np.ndarray);
            - No atributo img_shape: Tupla com dimensões das imagens;
            - No atributo dice: Valor do dice entre máscaras de teste e predições.

        :param gt_imgs: Máscaras de teste.
        :type test_images: numpy.ndarray
        :param pred_folder: Path para o diretório outputs_prod da execução n, onde estão salvas as predições.
        :type pred_folder: str
        :param new_size: Novo size das imagens.
        :type new_size: int

        '''
        self.gt_imgs = gt_imgs
        self.pred_folder = pred_folder
        self.new_size = new_size

        list_pred_imgs = sorted(glob.glob(f"{self.pred_folder}")) 
        self.pred_imgs, self.img_shape = Dataset.load_images_array(self, img_list=list_pred_imgs,original_size=self.new_size, new_size=self.new_size)

        self.dice = self.dice_coef(y_true=self.gt_imgs, y_pred=self.pred_imgs)
        print(f"Coeficiente Dice: {self.dice}")
        self.df = None

    def dice_coef(self, y_true:np.ndarray, y_pred:np.ndarray):
        '''
        Calcula o dice entre as máscaras de teste e as predições.

        :param y_true: Máscaras das imagens de teste.
        :type y_true: numpy.ndarray
        :param y_pred: Predições do modelo.
        :type y_pred: numpy.ndarray

        :return: Coeficiente Dice
        :rtype: float
        '''
        y_true_f = keras.backend.flatten(y_true) 
        y_pred_f = keras.backend.flatten(y_pred) 
        intersection = keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + \
            keras.backend.epsilon()) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + keras.backend.epsilon())  
    
    def save_dice(self, adress:str):
        '''
        Salva o valor do dice na pasta da execução n.

        :param adress: Path para arquivo .txt do dice.
        :type adress: str
        '''
        with open(adress, "w") as file:
            file.write(f"Dice: {self.dice}")

    def generate_csv_dice(self, n_all_folders:int, save_report, title:str):
        '''
        Cria um arquivo .csv no diretório do conjunto de execuções com o valor de todos os dices dos n_folds.
        Alimenta o atributo df com os mesmos dados do csv.

        :param n_all_folders: Número total de execuções no conjunto.
        :type n_all_folders: int
        :param save_report: Objeto da classe SaveReport para organização dos arquivos e diretórios.
        :type save_report: labic_images_segmentation.SaveReport
        :param title: Título do arquivo .csv. Recomenda-se indicar modelo, backbone, entre outras informações relevantes.
        :type title: str
        '''
        all_dice = []
        for i in range(n_all_folders+1):
            with open(f"{save_report.exec_folder_name}/fold_{i}/dice_fold_{i}.txt", "r") as file:
                content = file.read()
                dice = float(content[6:])
                all_dice.append(dice)
        index = [n for n in range(n_all_folders+1)] + ['Mean','Std-Dev', 'Median', 'Max', 'Min']
        values = all_dice + [np.average(all_dice), np.std(all_dice), np.median(all_dice), np.max(all_dice), np.min(all_dice)]
        all_dice_dict = {"Index":index, title:values}
        self.df = pd.DataFrame(all_dice_dict)
        self.df.to_csv(f"{save_report.exec_folder_name}/{title}.csv")
        print(f"Arquivo {title}.csv gerado com sucesso.")

    def generate_graphic(self, epochs:int, segment, save_report, graphic_type:str):
        '''
        Gera os gráficos com dados do History. Armazena as imagens em formato png.
        
        :param epochs: Número de épocas.
        :type epochs: int
        :param segment: Objeto da classe SegmentationModel.
        :type segment: labic_images_segmentation.SegmentationModel
        :param save_report: Objeto da classe SaveReport.
        :type save_report: labic_images_segmentation.SaveReport
        :param graphic_type: Tipo do gráfico ("iou_score" ou "loss").
        :type graphic_type: str
        '''
        if graphic_type=="iou_score":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = [epoch for epoch in range(epochs)], y = segment.history.history['val_iou_score'],
                        mode = 'lines', name = "Validation Iou Score", line = {'color': '#00AD5A'}))
            fig.add_trace(go.Scatter(x = [epoch for epoch in range(epochs)], y = segment.history.history['iou_score'],
                            mode = 'lines', name = "Train Iou Score", line = {'color': '#1D6DD8'}))
            
            fig.update_layout(title_text='Iou Score per Epoch', title_x=0.5,\
                            xaxis_title='Epochs', yaxis_title='Iou Score',\
                            height = 450, width = 800, font={'size':10})
            fig.show()
            fig.write_image(f"{save_report.n_fold_folder_name}/iou_score_report.png")

        if graphic_type=="loss":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = [epoch for epoch in range(epochs)], y = segment.history.history['loss'],
                            mode = 'lines', name = "Train Loss", line = {'color': '#D65200'}))

            fig.add_trace(go.Scatter(x = [epoch for epoch in range(epochs)], y = segment.history.history['val_loss'],
                            mode = 'lines', name = "Validation Loss", line = {'color': '#B40808'}))
            
            fig.update_layout(title_text='Loss per Epoch', title_x=0.5,\
                            xaxis_title='Epochs', yaxis_title='Loss',\
                            height = 450, width = 800, font={'size':10})
            fig.show()
            fig.write_image(f"{save_report.n_fold_folder_name}/loss_report.png")



