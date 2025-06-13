import sys
import os, ast
import yaml, json
import numpy as np
import wandb



import tensorflow as tf
# Import Intel Extension for TensorFlow for XPU support
import intel_extension_for_tensorflow as itex
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications.xception import preprocess_input as pp_input

# user-defined functions (from utils.py)
from dataset_info import datasetInfo
from datagen import DirectoryDataGenerator
from customcallbacks import ValCallback
from schedulers import StepLearningRateScheduler
from utils import get_flops
from models import construct_model

tf.compat.v1.experimental.output_all_intermediates(True)

# ######################################### PROCESSING DATASET DIRECTORY INFO ####################################### #
"""  Function for pre-processing directory information """
def process_dir(rootdir, dataset, model_name):

    dataset_dir = rootdir + dataset
    working_dir = os.path.dirname(os.path.realpath(__file__))
    train_data_dir = '{}/train/'.format(dataset_dir)
    val_data_dir = '{}/val/'.format(dataset_dir)
    if not os.path.isdir(val_data_dir):
        val_data_dir = '{}/test/'.format(dataset_dir)

    output_model_dir = '{}/TrainedModels/{}'.format(working_dir, model_name)
    metrics_dir = '{}/Metrics/{}'.format(working_dir, model_name)

    nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)]) # number of images used for training, including "other" action
    nb_val_samples = sum([len(files) for r, d, files in os.walk(val_data_dir)]) # number of images used for validation
    validation_steps = validation_freq

    return dataset_dir, train_data_dir, val_data_dir, output_model_dir, metrics_dir, nb_train_samples, validation_steps


""" Function for assigning varibales from config.yaml file """
all_vars = []
def assign_variables(dictionary, prefix=''):
    for key, value in dictionary.items():
        variable_name = key
        if isinstance(value, dict):
            assign_variables(value, variable_name + '_')
        else:
            globals()[variable_name] = value
            all_vars.append(variable_name)

# ######################### LOAD CONFIGURATIN AND OVERWRITE WITH CONSOLE PARAMTERS ################################## #

"""  Load and assign variables from the config file   """
dataset_name = sys.argv[sys.argv.index('dataset') + 1] if 'dataset' in sys.argv else None
try:
    param_dir = "./configs/config_" + dataset_name +".yaml"
except:
    print('Please provide a valid dataset name under the dataset argument; Example command ---> python hofi/train.py dataset Aircraft')
    sys.exit(1)

with open(param_dir, 'r', encoding='utf-8') as file:
    param = yaml.load(file, Loader = yaml.FullLoader)
print('Loading Default parameter configuration: \n', json.dumps(param, sort_keys = True, indent = 3))

# Assign 
assign_variables(param)


"""  Check and override with console paramaters  """
if len(sys.argv) > 2:
    total_params = len(sys.argv)
    for i in range(1, total_params, 2):
        var_name = sys.argv[i]
        new_val = sys.argv[i+1]
        try:
            exec("{} = {}".format(var_name, new_val))
        except:
            exec("{} = '{}'".format(var_name, new_val))

"""  << Fetcing directory info >>  """ 
dataset_dir, train_data_dir, val_data_dir, output_model_dir, metrics_dir, nb_train_samples, validation_steps = process_dir(rootdir, dataset, model_name)
nb_classes = datasetInfo(dataset)
print('\n Dataset Location --> ', dataset_dir, '\n', 'no_of_class --> ', nb_classes)
# print(train_data_dir, val_data_dir)

print('_________________ Wandb_Logging _______________ : ', wandb_log)
print(' ')

"""  << Intel XPU Device Configuration >>  """

def configure_intel_xpu():
    """配置Intel XPU设备"""
    print("=" * 60)
    print("Intel XPU 设备配置")
    print("=" * 60)

    # 检查Intel Extension for TensorFlow
    try:
        print(f"Intel Extension for TensorFlow版本: {itex.__version__}")
        print("✓ Intel Extension for TensorFlow已加载")
    except Exception as e:
        print(f"✗ Intel Extension for TensorFlow加载失败: {e}")
        return False, None

    # 检查物理设备
    physical_devices = tf.config.list_physical_devices()
    print(f"可用物理设备: {[d.name for d in physical_devices]}")

    # 检查XPU设备
    xpu_devices = tf.config.list_physical_devices('XPU')
    gpu_devices = tf.config.list_physical_devices('GPU')

    if xpu_devices:
        print(f"✓ 检测到 {len(xpu_devices)} 个XPU设备:")
        for i, device in enumerate(xpu_devices):
            print(f"  XPU:{i} - {device.name}")

        # 配置XPU内存增长
        try:
            for xpu in xpu_devices:
                tf.config.experimental.set_memory_growth(xpu, True)
            print("✓ XPU内存增长配置成功")
        except Exception as e:
            print(f"⚠ XPU内存配置警告: {e}")

        return True, 'XPU'

    elif gpu_devices:
        print(f"✓ 检测到 {len(gpu_devices)} 个GPU设备:")
        for i, device in enumerate(gpu_devices):
            print(f"  GPU:{i} - {device.name}")

        # 配置GPU内存增长
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU内存增长配置成功")
        except Exception as e:
            print(f"⚠ GPU内存配置警告: {e}")

        return True, 'GPU'
    else:
        print("ℹ 未检测到XPU或GPU设备，使用CPU")
        return True, 'CPU'

def get_device_strategy(device_type, gpu_id):
    """获取设备策略"""
    if device_type == 'XPU':
        if gpu_id >= 0:
            device_name = f'/XPU:{gpu_id}'
            print(f"✓ 使用指定的XPU设备: {device_name}")
        else:
            device_name = '/XPU:0'
            print(f"✓ 使用默认XPU设备: {device_name}")
        return device_name
    elif device_type == 'GPU':
        if gpu_id >= 0:
            device_name = f'/GPU:{gpu_id}'
            print(f"✓ 使用指定的GPU设备: {device_name}")
        else:
            device_name = '/GPU:0'
            print(f"✓ 使用默认GPU设备: {device_name}")
        return device_name
    else:
        print("✓ 使用CPU设备")
        return '/CPU:0'

def create_optimized_session_config(device_type, gpu_utilisation):
    """创建优化的会话配置"""
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    config.inter_op_parallelism_threads = 0  # 使用所有可用核心
    config.intra_op_parallelism_threads = 0  # 使用所有可用核心

    if device_type in ['XPU', 'GPU']:
        # 对于XPU和GPU，设置内存分配
        if device_type == 'GPU':
            config.gpu_options.per_process_gpu_memory_fraction = gpu_utilisation
            config.gpu_options.allow_growth = True
        print(f"✓ {device_type}会话配置完成，内存使用率: {gpu_utilisation}")
    else:
        print("✓ CPU会话配置完成")

    return config

# 配置Intel XPU
has_accelerator, device_type = configure_intel_xpu()
device_strategy = get_device_strategy(device_type, gpu_id)

# 创建会话配置
session_config = create_optimized_session_config(device_type, gpu_utilisation)
sess = tf.compat.v1.Session(config=session_config)

# 兼容性设置
if device_type == 'GPU':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

tf.compat.v1.disable_eager_execution()

print("=" * 60)
print(f"设备配置完成 - 使用: {device_type}")
print("=" * 60)

#################################### LOGGING, BUILDING & TRAINING MODEL #############################################
# ============== Wandb LOGGING Parameters ======================= #
if wandb_log:
    wandb.login(key="4ca920c713ce2c8409d420dac4cc4361075f47ad") # WandB API key
    wrun=wandb.init(
        project = wb_proj_name,
        name = model_name if run_name == 'None' else run_name,
        config = {
                "epochs": epochs,
                "batch_size": batch_size,
                "nb_classes" : nb_classes,
                "lr": lr,
                "validation_steps" : validation_steps,
                "checkpoint_freq" : checkpoint_freq,
                "completed_epochs" : completed_epochs,
                "gcn_outfeat_dim" : gcn_outfeat_dim,
                "gat_outfeat_dim" : gat_outfeat_dim,
                "dropout_rate" : dropout_rate,
                "l2_reg" : l2_reg,
                "attn_heads" : attn_heads,
                "appnp_activation" : appnp_activation,
                "gat_activation" : gat_activation,
                "concat_heads" : concat_heads,
                "reduce_lr_bool" : reduce_lr_bool,
                "backbone": backbone,
                "freeze_backbone": freeze_backbone,
                "GNN_layer1": gnn1_layr,
                "GNN_layer2": gnn2_layr,
                "alpha": alpha,
                "pool_size": pool_size,
                "input_sh" : image_size,
                }
        )
    pconfig = wandb.config

# ============ Building model ============== #
model = construct_model(
    name = model_name,
    pool_size = pool_size,
    ROIS_resolution = ROIS_resolution,
    ROIS_grid_size = ROIS_grid_size,
    minSize = minSize,
    alpha = alpha,
    nb_classes = nb_classes,
    batch_size = batch_size,
    input_sh = image_size,
    gcn_outfeat_dim = gcn_outfeat_dim,
    gat_outfeat_dim = gat_outfeat_dim,
    dropout_rate = dropout_rate,
    l2_reg = l2_reg,
    attn_heads = attn_heads,
    appnp_activation = appnp_activation,
    gat_activation = gat_activation,
    concat_heads = concat_heads,
    backbone = backbone,                # default : Xception', the backbone name is case-sensitive
    freeze_backbone = freeze_backbone,  # Option to Freeze the backbone while training appended GNN framework
    gnn1_layr = gnn1_layr,
    gnn2_layr = gnn2_layr,
    track_feat = track_feat,
    )

# Giga-flops calculation or Train
if calflops:
    r = get_flops( model, tf.compat.v1.placeholder('float32', shape=(1, image_size[0], image_size[1], image_size[2])) ) 
    print('~~~~~~~~ Total FLOPs --> {} | Giga-FLOPs --> {} ~~~~~~~~'.format(r, r/10**9))

    # print model summary
    if summary:
        model.summary()
else:
    # Initializing the model
    outputs = model(model.base_model.input)
    
    # print model summary
    if summary:
        model.summary()
        
    # Building the (functional) model
    model = Model(inputs = model.input, outputs = outputs)
    
    
    callbacks = []
    # ============== Inititalizing Data Generation for training samples ================== % 
    train_dg = DirectoryDataGenerator(
        base_directories = [train_data_dir], 
        augmentor = True, 
        target_sizes = image_size[:2], 
        preprocessors = pp_input, 
        batch_size = batch_size, 
        shuffle = True, 
        channel_last = True, 
        verbose = 1, 
        hasROIS = False
        )
    
    # ========== Inititalizing Data Generation for Validation/test samples ============== %
    val_dg = DirectoryDataGenerator(
        base_directories = [val_data_dir], 
        augmentor = None, 
        target_sizes = image_size[:2], 
        preprocessors = pp_input, 
        batch_size = batch_size, 
        shuffle = False, 
        channel_last = True, 
        verbose = 1, 
        hasROIS = False
        )
    
    # Setting FILENAME for Saving model checkpoint
    if not load_model:
        try:
            filename_parts = [
                wrun.id, dataset, backbone, f"Bs-{batch_size}",
                f"init_lr-{lr}", "epoch-{:03d}", "lr-{:.6f}", "valAcc-{:.4f}.h5"
            ]
            safe_filename = "-".join(str(p) for p in filename_parts)
            checkpoint_path = os.path.join(output_model_dir, safe_filename)
        except NameError:
            filename_parts = [
                dataset, backbone, f"Bs-{batch_size}",
                f"init_lr-{lr}", "epoch-{:03d}", "lr-{:.6f}", "valAcc-{:.4f}.h5"
            ]
            safe_filename = "-".join(str(p) for p in filename_parts)
            checkpoint_path = os.path.join(output_model_dir, safe_filename)

    # =========== Custom CALLBACKS to record Validation metrics ============= #
    callbacks.append(
        ValCallback(
            val_dg, 
            validation_steps, 
            metrics_dir + model_name,
            wandb_log,
            save_model,
            checkpoint_path,
            save_best_only,
            checkpoint_freq
            )
        )
    
    if reduce_lr_bool == True:
        # Define the epochs at which to reduce the learning rate and create lr scheduler callback
        schedule_epochs = [50, 100, 150]
        lr_schedule = StepLearningRateScheduler(schedule_epochs, factor = 0.1)
        callbacks.append(lr_schedule)
    
    
    # =======================  Building engine ======================== #
    if load_model:
        print('_____________ Checkpoint path to Load Pretrained Model _____________ :', checkpoint_path)
        model.load_weights(checkpoint_path)
    
    # Use TensorFlow 1.x compatible optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = lr)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    ##################################################################
    # ################## Training Model ############################ #
    ##################################################################
    steps_per_epoch = nb_train_samples // batch_size
    model.fit(
        train_dg, 
        steps_per_epoch = steps_per_epoch, 
        initial_epoch = completed_epochs,  
        epochs = epochs, 
        callbacks = callbacks
        ) #train and validate the model
    
    if wandb_log:
        wrun.finish()
