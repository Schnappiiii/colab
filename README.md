# Battery aging trajectory prediction with deep learning

Excessive battery aging can lead to disasters such as explosions, so accurate prediction of battery aging is crucial. Moreover, battery aging tests are costly and time-consuming. The aim of this work is to deploy deep learning models on more complicated cases and use as little input data as possible.

## Datasets
- MIT/Stanford battery dataset (public)
  
  Original dataset: [Data-driven prediction of battery cycle life before capacity degradation](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204). Processed by Thomas in this [paper](https://www.sciencedirect.com/science/article/abs/pii/S2352152X23022399).

  ├── [datasets/Stanford_battery_dict_len8.pkl](./datasets/Stanford_battery_dict_len8.pkl)

  The final processed data:

  ├── [datasets/Stanford_battery_f8_cubicSpline_dict.pkl](./datasets/Stanford_battery_f8_cubicSpline_dict.pkl)

- FTM (Chair of Automotive Technology) battery dataset (not public)
  
  Processed by Thomas in this [paper](https://www.sciencedirect.com/science/article/abs/pii/S2352152X23022399).

  ├── [datasets/TUM_battery_dict.pkl](./datasets/TUM_battery_dict.pkl).

  The final processed data:

  ├── [datasets/TUM_battery_f6_cubicSpline_dict.pkl](./datasets/TUM_battery_f6_cubicSpline_dict.pkl)
  
### Synthetic datasets (MIT/Stanford battery dataset)
1. Original batteries + scaled batteries with EFC > 1000 (total 164 batteries) for mixed battery data: 
   
   ├── [datasets/Stanford_battery_all_1000synthetic_dict1.pkl](./datasets/Stanford_battery_all_1000synthetic_dict1.pkl)
2. Scaled battery aging curves + exponential function curves for pre-training and fine tuning (total 145 batteries):
   
   ├── [datasets/Stanford_battery_exp_synthetic_dict.pkl](./datasets/Stanford_battery_exp_synthetic_dict.pkl)
3. Original batteries with EFC > 1000 + scaled batteries with EFC > 1000 (total 98 batteries) for training models under battery EFC > 1000:
   
   ├── [datasets/Stanford_battery_1000synthetic_dict.pkl](./datasets/Stanford_battery_1000synthetic_dict.pkl)

### Synthetic datasets (FTM battery dataset)
1. Original batteries + scaled batteries (29+29 = total 58 batteries) for mixed battery data:
   
   ├── [datasets/TUM_battery_mix_dict.pkl](./datasets/TUM_battery_mix_dict.pkl)
   
2. Mixed batteries with EFC < 1500 and scaled batteries with EFC >= 1500:
   
   ├── [datasets/TUM_battery_mix_eol_dict1.pkl](./datasets/TUM_battery_mix_eol_dict1.pkl)
   
   ├── [datasets/TUM_battery_mix_eol_dict2.pkl](./datasets/TUM_battery_mix_eol_dict2.pkl)
   
### Age days datasets (MIT/Stanford and FTM battery dataset)
Input is age day unit, not EFC.
1. ├── [datasets/Stanford_age_days_dict.pkl](./datasets/Stanford_age_days_dict.pkl) (not performed in this experiment)

2. ├── [datasets/TUM_age_days_dict.pkl](./datasets/TUM_age_days_dict.pkl) (granularity 1: [0,1,2,3,4,5,6,7,8,9,...])

3. ├── [datasets/TUM_age_days_more_dict.pkl](./datasets/TUM_age_days_more_dict.pkl) (granularity 0.1: [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,...], using sampling_frequency: 10 in hyperparameters.yaml = granularity 1, so [datasets/TUM_age_days_dict.pkl](./datasets/TUM_age_days_dict.pkl) can be ignored.)

### Other datasets (MIT/Stanford and FTM battery dataset)
Other methods for data processing were also conducted. However, they were not shown in this work since the prediction results were not well. 
├── [other_datasets](./other_datasets) 

<b> Please don't forget to copy datasets from ‘datasets’ folder to the individual model 'dataset' folder before training <b>, for example, [datasets/Stanford_battery_dict_len8.pkl](./datasets/Stanford_battery_dict_len8.pkl) --> [ae_cnn_rnn_attention/datasets/Stanford_battery_dict_len8.pkl](./ae_cnn_rnn_attention/datasets/Stanford_battery_dict_len8.pkl)

## Models
Because this work only focused on DeAE-CNN-GRU-Attention and Informer-CNN-LSTM these two new models, the comments in the code script of CNN-GRU-Attention and LSTM-Autoencoder are not comprehensive (codes are similar to DeAE-CNN-GRU-Attention model).
1. [ae_cnn_rnn_attention](./ae_cnn_rnn_attention) denotes DeAE-CNN-GRU-Attention model. But you can change different models combined with Denoising Autoencoder in [ae_cnn_rnn_attention/hyperparameters.yaml](./ae_cnn_rnn_attention/hyperparameters.yaml), namely, ae_cnn_rnn_attention, ae_rnn, ae_cnn_rnn, ae_cnn_rnn_attention, ae_transformer_encoder. CNN type can be chosen between vanilla and residual forms. Model ae_transformer_encoder implements this [paper](https://ieeexplore.ieee.org/document/9714323) and using fully connected layers to construct Denoising Autoencoder.
2. [informer](./informer) denotes [Informer model](https://arxiv.org/abs/2012.07436). Available model combinations in [informer/hyperparameters.yaml](./informer/hyperparameters.yaml) are informer, informerstack and informer_cnn_rnn.
3. [cnn_rnn_attention](./cnn_rnn_attention) denotes CNN-GRU-Attention model. Available model combinations in [cnn_rnn_attention/hyperparameters.yaml](./cnn_rnn_attention/hyperparameters.yaml) are rnn, cnn_rnn, fpn_attention (Feature Pyramid Network + Attention), cnn_rnn_attention, fpn_rnn_attention and transformer_encoder. Model transformer_encoder implements this [paper](https://www.mdpi.com/1996-1073/16/17/6328) and using residual neural network to construct Denoising Network. Model cnn_rnn_attention implements this [paper](https://journals.sagepub.com/doi/full/10.1177/17483026221130598).
4. [rnn_autoencoder](./rnn_autoencoder) denotes RNN-Autoencoder model. Available model combinations in [rnn_autoencoder/hyperparameters.yaml](./rnn_autoencoder/hyperparameters.yaml) are rnn_autoencoder, rnn_autoencoder1 (step by step prediction in Decoder part), rnn_autoencoder_cnn_attention and rnn_autoencoder_transencoder (RNN-Autoencoder + Transformer Encoder). Model rnn_autoencoder implements this [paper](https://www.sciencedirect.com/science/article/pii/S0378775321005528). Model rnn_autoencoder1 implements this [paper](https://www.frontiersin.org/articles/10.3389/fenrg.2022.1093667/full).

5. [Binary classification](./classification.ipynb) denotes binary classification for two stage prediction. KNN, Random forest, XGBoost and LSTM classification models.

All RNNs can be replaced by LSTM, GRU or vanilla RNNs.

## Hyperparameters (hyperparameters.yaml)
 - data_truth_path means original data for drawing (keep this). data_process_path1 means processed data (interpolation) for calculating the loss on full data set in last step (keep this). Changing data_process_path for different processed data (e.g., interpolation, synthesis). 
- Scaler (preprocessing) can be selected between MinMaxScaler and StandardScaler (MinMaxScaler is better for battery aging predcition). 
- max_efcs refers to only training batteries whose battery life cycle is less than max_efcs EFC (two stage prediction). 
- sampling_frequency, seq_len, pred_len: If we want to input 280 EFC and output 200 EFC under sampling_frequency 2, seq_len is then 280/2=140, pred_len is 200/2=100.

## Result
Build model directory to save model, result images and so on.
```
├── model_result
    ├── model_name
        ├── experiment1
            ├── image
                ├── train_prediction
                ├── validation_prediction
                ├── test_prediction
                ├── full_original_prediction
                ├── new_data_prediction (if necessary)
                - training_validation_loss.png
            - hyperparameters.yaml
            - experiment_parameters.yaml
            - MinMaxScaler.pkl
            - checkpoint.pth
            - training_prediction_summary_dict.pkl
            - validation_prediction_summary_dict.pkl
            - test_prediction_summary_dict.pkl
            - full_original_prediction_summary_dict.pkl
            - summery_dict.pkl
        - experiment2
        - ...
    ├── sensitivity_analysis_result (if necessary)
```


============================
## Getting started
Notice: if you use Google Colab then don't need to create and set an Anaconda environment.

1. If you save this folder in the Google Dive, you need first connected Colab to Dive. (not recommend, because sometimes Colab will automatically disconnected to Drive. 😂)
```
 from google.colab import drive
 drive.mount('/content/drive')
```
The other simple way is to zip the file (.zip) before uploading it to Colab. Open [run.ipynb](./run.ipynb), in the colab interface go to the left side and click on file, then drag and drop the compressed file into the window. After that,
```
 !unzip 
 !python
 !zip -r result.zip
 Downloading...
```

Recommend use GPU since training model is time comsuming. 



