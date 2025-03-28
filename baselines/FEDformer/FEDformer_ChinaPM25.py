import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from .arch import FEDformer
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mse, masked_mae


CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "FEDformer model configuration"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "ChinaPM2.5"
CFG.DATASET_TYPE = "Electricity Transformer Temperature"
CFG.DATASET_INPUT_LEN = 72
CFG.DATASET_OUTPUT_LEN = 72
CFG.GPU_NUM = 1
# CFG.RESCALE = False

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 123
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "FEDformer"
CFG.MODEL.ARCH = FEDformer
NUM_NODES = 1316
CFG.MODEL.PARAM = EasyDict(
    {
    "enc_in": NUM_NODES,                              # num nodes
    "dec_in": NUM_NODES,
    "c_out": NUM_NODES,
    "seq_len": CFG.DATASET_INPUT_LEN,           # input sequence length
    "label_len": CFG.DATASET_INPUT_LEN/2,       # start token length used in decoder
    "pred_len": CFG.DATASET_OUTPUT_LEN,         # prediction sequence length\
    "d_model": 64,
    "version": "Fourier",                       # for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]
    "moving_avg": 24,                           # window size of moving average
    "n_heads": 8,
    "e_layers": 2,                              # num of encoder layers
    "d_layers": 1,                               # num of decoder layers
    "d_ff": 64,
    "dropout": 0.05,
    "output_attention": False,
    "embed": "timeF",                           # [timeF, fixed, learned]
    "mode_select": "random",                    # for FEDformer, there are two mode selection method, options: [random, low]
    "modes": 64,                                # modes to be selected random 64
    "base": "legendre",                         # mwt base
    "L": 3,                                     # ignore level
    "cross_activation": "tanh",                 # mwt cross atention activation function tanh or softmax
    "activation": "gelu",
    "num_time_features": 3,                     # number of used time features
    "time_of_day_size": 24,
    "day_of_week_size": 7,
    "day_of_month_size": 31,
    "day_of_year_size": 366
    }
)
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.0001
}

# ================= train ================= #
CFG.TRAIN.NUM_EPOCHS = 50
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 10
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

# ================= evaluate ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [12, 24, 48, 96, 192, 288, 336]
