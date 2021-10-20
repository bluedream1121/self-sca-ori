import os, wandb, argparse

def get_train_config():
    ## 1. hyperparameters
    parser = argparse.ArgumentParser(description='Self-supervised learning of image scale and orientation estimation.')

    parser.add_argument('--output_ori', type=int, default=36, help="orientation histogram size; default: 36")
    parser.add_argument('--output_sca', type=int, default=13, help="scale histogram size; default: 13")
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help='0.9, 0.99, 0.999')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--dataset_type', type=str, default="ppa_ppb", help="ppa: patchposeA, ppb: patchposeB, ppa_ppb: patchposeA + patchposeB")
    parser.add_argument('--load', type=str, default=False, help="Load a trained model.")
    parser.add_argument('--branch', type=str, default='ori', help='alter, sca, ori; alter is batch-alternation of sca/ori. ')
    parser.add_argument('--softmax_temperature', type=float, default=20, help='softmax temperature. ')

    args = parser.parse_args()
    print(args)

    ## 2. wandb init
    os.environ["WANDB_MODE"] = "dryrun"
    
    wandb.init()
    wandb.config.update(args, allow_val_change=True)

    return args

def get_test_config():
    ## hyperparameters
    parser = argparse.ArgumentParser(description='test code.')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--test_set', type=str, default='test')
    parser.add_argument('--dataset_type', type=str, default='ppa_ppb', help='ppa_ppb, hpa.')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--output_file_ori', type=str, default='', help='e.g. _1118_ori.txt')
    parser.add_argument('--output_file_sca', type=str, default='', help='e.g. _1118_sca.txt')
    args = parser.parse_args()

    return args
