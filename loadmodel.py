import os
import torch
import numpy as np
import time
from downstream_tasks.util.misc import load_model
from downstream_tasks.engine_finetune import evaluate
from downstream_tasks.models_vit import VisionTransformer
from src.datasets.hca_sex_datasets import make_hca_sex


import argparse

import multiprocessing as mp

import pprint
import yaml

from src.utils.distributed import init_distributed
from src.train import main as app_main
from torchinfo import summary

class Config:
    # Define the Config class with its relevant fields here
    pass
def loading_model(args):
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))

    # Configurer le dispositif (CPU ou GPU)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fixer la graine pour la reproductibilité
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    
    

    # Initialiser le modèle
    model = VisionTransformer(
        args,
        model_name=args.model_name,
        attn_mode=args.attn_mode,
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
        device=device,
        add_w=args.add_w
    )
    
    # Charger le checkpoint pré-entraîné
    if args.finetune:
        checkpoint = torch.load(args.finetune,weights_only=False, map_location='cpu')
        print(f"Loading pre-trained checkpoint from: {args.finetune}")
        


        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        # Ajuster les clés du checkpoint au modèle
        new_checkpoint_model = {}
        for key in checkpoint_model.keys():
            new_key = key.replace('module.', 'encoder.')
            new_checkpoint_model[new_key] = checkpoint_model[key]

        msg = model.load_state_dict(new_checkpoint_model, strict=False)
        print("Checkpoint loading:", msg)

    # Envoyer le modèle sur le dispositif
    model.to(device)
    model.eval()
    return model 
    


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
        crop_size = 450, 160#450,490 #450,490
        pred_depth= 12
        pred_emb_dim = 768
        gradient_checkpointing= True 
        add_w= 'origin'
        attn_mode= 'normal'
    
    model=loading_model(Argseval)
    #random_tensor = torch.randn( 1,1, 450, 160)
    

    print( 'ouuhh let s explore more about the result ! ')
    batch_size = 1
    channels = 1
    height = 450
    width = 160

    # Generate random tensor
    random_tensor = torch.randn(batch_size, channels, height, width)

    
    summary( model, input_size = ( 1,1,450,168))
    breakpoint()
    output= model(random_tensor)
    print( 'yes ! we passed the model and we the shape of the out put is', output ) 
    