import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from .arch import iTransformer
from basicts.metrics import masked_mse, masked_mae


CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "DLinear model configuration"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "ChinaWind"
CFG.DATASET_TYPE = "Weather"
CFG.DATASET_INPUT_LEN = 60
CFG.DATASET_OUTPUT_LEN = 30
CFG.GPU_NUM = 1
# CFG.RESCALE = False

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 120
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "iTransformer"
CFG.MODEL.ARCH = iTransformer
NUM_NODES = 396
CFG.MODEL.PARAM = EasyDict(
    {
        "enc_in": NUM_NODES,                        # num nodes
        "dec_in": NUM_NODES,
        "c_out": NUM_NODES,
        "seq_len": CFG.DATASET_INPUT_LEN,
        "label_len": CFG.DATASET_INPUT_LEN / 2,       # start token length used in decoder
        "pred_len": CFG.DATASET_OUTPUT_LEN,         # prediction sequence length
        "factor": 3, # attn factor
        "d_model": 64,
        "moving_avg": 25,                           # window size of moving average. This is a CRUCIAL hyper-parameter.
        "n_heads": 4,
        "e_layers": 2,                              # num of encoder layers
        "d_layers": 1,                              # num of decoder layers
        "d_ff": 64,
        "distil": True,
        "sigma" : 0.2,
        "dropout": 0.2,
        "freq": 'd',
        "use_norm" : True,
        "output_attention": False,
        "embed": "timeF",                           # [timeF, fixed, learned]
        "activation": "gelu",
        "root_path": "./datasets/ChinaWind"
    }
)
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]    # [raw_data, time_of_day, day_of_week, day_of_month, day_of_year]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.0005,
    "weight_decay": 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 25],
    "gamma": 0.5
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
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 32
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
CFG.TEST.DATA.DIR = 'datasets/' + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

# ================= evaluate ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [24]
