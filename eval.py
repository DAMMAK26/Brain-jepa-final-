from loadmodel import loading_model

import os
import torch
import numpy as np
import time
from downstream_tasks.util.misc import load_model
from downstream_tasks.engine_finetune import evaluate
from src.datasets.hca_sex_datasets import make_hca_sex

import argparse

import multiprocessing as mp

import pprint
import yaml

from src.utils.distributed import init_distributed
from src.train import main as app_main


def evalmodel(model, args):
  
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))

    # Configurer le dispositif (CPU ou GPU)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
      # Fixer la graine pour la reproductibilité
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

      # Charger les données
    if args.data_make_fn == 'hca_sex':
        data_fn = make_hca_sex
    else:
        raise ValueError(f"Data function {args.data_make_fn} not implemented!")
    
    # Envoyer le modèle sur le dispositif
    model.to(device)
    model.eval()

    print( "the data function",data_fn)
    _, data_loader_val, data_loader_test, _, valid_dataset, test_dataset = data_fn(
        batch_size=args.batch_size,
        pin_mem=args.pin_mem,
        num_workers=args.num_workers,
        collator=None,
        #world_size=1,
        #rank=0,
        drop_last=False,
        #data_split=[0.6, 0.2, 0.2],
        processed_dir=args.data_dir,
        use_normalization=args.use_normalization,
        label_normalization=args.label_normalization,
        downsample=args.downsample
    )

   

    print(f"Validation dataset length: {len(valid_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}") 
    # Évaluation sur le jeu de validation
    print("Evaluating on validation dataset...")
    val_stats = evaluate(args, data_loader_val, model, device, args.task)

    # Évaluation sur le jeu de test
    print("Evaluating on test dataset...")
    test_stats = evaluate(args, data_loader_test, model, device, args.task)

    if args.task == 'classification':
        print(f"Validation Accuracy: {val_stats['acc1']:.2f}%")
        print(f"Test Accuracy: {test_stats['acc1']:.2f}%")
    else:
        print(f"Validation MSE: {val_stats['loss']:.3f}, R2: {val_stats['r2']:.3f}")
        print(f"Test MSE: {test_stats['loss']:.3f}, R2: {test_stats['r2']:.3f}")


if __name__ == "__main__":
    class Argseval:
        device = "cuda"
        seed = 42
        batch_size = 64
        pin_mem = True
        num_workers = 12
        data_make_fn = 'hca_sex'
        data_dir = "data/processed/hca_lifespan"
        use_normalization = True
        label_normalization = False
        downsample = None
        finetune = "E:/recherche/brain/brain-jepa/Brain-JEPA-main/Brain-JEPA-main/checkpoint.pth"
        model_name = "vit_base"
        attn_mode = "self"
        nb_classes = 2
        global_pool = True
        add_w = None
        task = 'classification'
        patch_size = 16
        crop_size = 450,160
        pred_depth= 12
        pred_emb_dim = 768
        gradient_checkpointing= False
    
    model=loading_model(Argseval)
    evalmodel(model, Argseval)

