from solver import Solver
import argparse
from datetime import datetime
import os

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--saved_model', type=str, default=os.path.join(os.getcwd(), '16-epoch-0.6548.pt'))
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--mode', type=str, default = 'train', choices=['train', 'test'])
    parser.add_argument('--lr', type=int, default = 1e-3)
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)

    # Directories.
    parser.add_argument('--img_dir', type=str, default=os.path.join(os.getcwd(),'cracky'))
    parser.add_argument('--save_dir', type=str, default=os.getcwd())
    parser.add_argument('--test_dir', type = str, default = os.path.join(os.getcwd(), 'test'))
    
    # Step size.
    parser.add_argument('--log_step', type=int, default = 100)
    parser.add_argument('--sample_step', type=int, default = 1000)
    parser.add_argument('--model_save_step', type=int, default = 10000)
    parser.add_argument('--lr_update_step', type=int, default = 1000)

    config = parser.parse_args()
    config.save_dir = os.path.join(config.save_dir, datetime.today().strftime('%m-%d--%H:%M'))
    solver = Solver(config)
    
    if config.mode=='train':
        if not os.path.isdir(config.save_dir):
            os.makedirs(config.save_dir)
        solver.train()
    
    elif config.mode=='test':
        solver.test()