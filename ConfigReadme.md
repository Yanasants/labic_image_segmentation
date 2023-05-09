### Config.json - Arquivo de parâmetros

* __n_fold__: Número da execução;
* __dataset_folder__: Diretório com dados de treino;
* __norm_imgs_folder__: Subdiretório com imagens de treino;
* __gt_folder__: Subdiretório com máscaras de treino;
* __output_folder__: Diretório com dados de teste;
* __test_imgs_folder__: Subdiretório com imagens de teste;
* __gt_test_folder__: Subdiretório com máscaras de teste;
* __ORIGINAL_SIZE__: Size (em pixels) original das imagens;
* __NEW_SIZE__: Novo size (em pixels) das imagens redimensionadas;
* __batch_size__: Tamanho do lote (batch-size);
* __epochs__: Número de épocas;
* __exec_folder_name__: Diretório do conjunto de execuções. [_Ex.:Exec_2023-05-09-11-38-45.654295_];
* __title__: Título para arquivo de report do conjunto de execuções. [_Ex.: Unet-resnet34_];
* __model__: Modelo para segmentação. [_'unet_original', 'unet', 'linknet'_];
* __backbone_name__: Backbone (dentro das opções do segmentation_models);
* __callback__: Callback. [_'none', 'early_stopping'_];
* __monitor__: Variável a ser monitorada quando utilizado o Early Stopping. [_+info: https://keras.io/api/callbacks/early_stopping/_];
* __verbose__: Modo de verbosidade quando utilizado o Early Stopping. [_+info: https://keras.io/api/callbacks/early_stopping/_];
* __mode__: Modo de ação do Early Stopping no treinamento. [_+info: https://keras.io/api/callbacks/early_stopping/_];
* __patience__: Número de épocas sem melhora após as quais o treinamento será interrompido, quando utilizado o Early Stopping;
