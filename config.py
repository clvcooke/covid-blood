import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='covid')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--batch_size', type=int, default=16,
                      help='# of images in each batch of data')
data_arg.add_argument('--exclusion', type=str, default=None,
                      help='Path of file containing list of files to exclude')

train_arg = add_argument_group('Training Params')
train_arg.add_argument('--epochs', type=int, default=50,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=1e-3,
                       help='Initial learning rate value')
train_arg.add_argument('--lr_patience', type=int, default=5,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--train_patience', type=int, default=1000,
                       help='Number of epochs to wait before stopping train')
train_arg.add_argument('--random_seed', type=int, default=0,
                       help='Random seed')
train_arg.add_argument('--fold_number', type=int, default=0,
                       help='Which fold to use for validation')
train_arg.add_argument('--model_name', type=str, default='densenet',
                       help='which architecture to use')
train_arg.add_argument('--pretrained_model', type=str2bool, default=True,
                       help='pretraining True or False')

misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--gpu_number', type=int, default=0,
                      help="Which GPU to use")
misc_arg.add_argument('--task', type=str, default='covid-class',
                      help='Which task to launch')
misc_arg.add_argument('--test_interval', type=int, default=5,
                      help="How often to evaluate the test set")
misc_arg.add_argument('--warm_start', type=str, default="",
                      help="Model to warm start with")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed