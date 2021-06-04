# TianChi AIEarth
<p align="justify">
This is the top 1 (B榜) solution for the TianChi AIEarth Contest (https://tianchi.aliyun.com/competition/entrance/531871/introduction?lang=zh-cn)

The three models are implemented in Pytorch for Nino prediction and currently support single-GPU training only.
Note that the model architecture of the SAConvLSTM and Conv-TT-LSTM are from the original papers, and we novelly modify the TimeSformer as STTransformer for spatio-temporal prediction.
</p>

# Reference
- SAConvLSTM\
Self-Attention ConvLSTM for Spatiotemporal Prediction (Lin et al., 2020) [https://ojs.aaai.org/index.php/AAAI/article/view/6819](https://ojs.aaai.org/index.php/AAAI/article/view/6819)
- Conv-TT-LSTM\
Convolutional Tensor-Train LSTM for Spatio-temporal Learning (Su et al., 2020) [https://arxiv.org/abs/2002.09131](https://arxiv.org/abs/2002.09131)
- TimeSformer\
Is Space-Time Attention All You Need for Video Understanding? (Bertasius et al., 2021) [https://arxiv.org/abs/2102.05095](https://arxiv.org/abs/2102.05095)
