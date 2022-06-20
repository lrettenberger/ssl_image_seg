from torch_cka import CKA
from pathlib import Path
from torchvision.models import resnet50

def calculate_cka(data,directory,model,ref_model):
        Path(f"{directory}/cka").mkdir(parents=True, exist_ok=True)
        
        data.shuffle = False
        # batch size should be small -> 2 models on gpu
        data.batch_size = data.batch_size // 4
        
        layers = []
        for name,weights in model.named_modules():
                if 'backbone.1' in name:
                        layers.append(name)

        cka = CKA(model, ref_model,
                model1_name="model",   # good idea to provide names to avoid confusion
                model2_name="ref_model",   
                model1_layers=layers,
                model2_layers=layers,
                device='cuda')

        cka.compare(data.test_dataloader()) # secondary dataloader is optional
        results = cka.export()  # returns a dict that contains model names, layer names
                                # and the CKA matrix
        cka.plot_results(save_path=f"{directory}/cka/cka.png")