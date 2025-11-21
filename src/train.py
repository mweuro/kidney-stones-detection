from argparse import ArgumentParser
from datetime import datetime
from xml.parsers.expat import model
import torch
from ultralytics import YOLO
import yaml


# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Automatic device selection
device = "0" if torch.cuda.is_available() else "cpu"
config['train']['device'] = device
config['validate']['device'] = device



if __name__ == '__main__':
    parser = ArgumentParser(
        prog="Kidney stone detection",
        description="Train and validate a YOLO model using specified configurations."
    )
    parser.add_argument('--model-name', 
                        type=str, 
                        default="yolov8n.pt",
                        help="Name of the YOLO model to use (default: yolov8n.pt)")
    parser.add_argument('--data',
                        type=str, 
                        default="data_preprocessed/data.yaml",
                        help="Path to the dataset YAML file (default: data_preprocessed/data.yaml)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name_train = f"train_{args.model_name.split('.')[0]}_{args.data.split('/')[0]}_{timestamp}"
    name_val = f"val_{args.model_name.split('.')[0]}_{args.data.split('/')[0]}_{timestamp}"
    
    model = YOLO(args.model_name)
    model.train(data=args.data, name=name_train, **config['train'])
    model.val(data=args.data, name=name_val, **config['validate'])