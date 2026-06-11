import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Per Kaggle
try:
    from models import get_resnet18
    from data_module import get_dataloaders, TriggerSetDataset
except ImportError:
    # Fallback/Cheat for raw Kaggle cell execution if files are inline
    pass

# =====================================================================
# 1. EVALUATION CORE FUNCTION
# =====================================================================
def evaluate_model(model, dataloader, device):
    """
    Computes top-1 accuracy for a given model and dataloader.
    Operates strictly under torch.no_grad() for maximum efficiency.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    if total == 0:  # Protezione anti ZeroDivisionError
        return 0.0
    return 100. * correct / total

# =====================================================================
# 2. STANDARD FINE-TUNING / RE-TRAINING ATTACKS
# =====================================================================
def run_ft_attack(base_model, attack_type, clean_train_loader, test_loader, trigger_loader, device, epochs=5): 
    """
    Executes Fine-Tuning (FT) and Re-Training (RT) variations on Last Layer (LL) or All Layers (AL).
    """
    # PyTorch-native clone alternativo a deepcopy (più stabile su Kaggle)
    model = get_resnet18(pretrained=False).to(device)
    model.load_state_dict(base_model.state_dict())
    
    criterion = nn.CrossEntropyLoss()
    
    # Configure parameter freezing and re-initialization based on attack taxonomy
    if attack_type in ["FTLL", "RTLL"]:
        # Freeze all layers except the classification head (fc)
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        if attack_type == "RTLL":
            # Re-initialize the classification head weights
            model.fc.reset_parameters()
        optimizer = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
    elif attack_type in ["FTAL", "RTAL"]:
        # Keep all layers trainable
        for param in model.parameters():
            param.requires_grad = True
        if attack_type == "RTAL":
            # Re-initialize the classification head weights while leaving features pretrained
            model.fc.reset_parameters()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    # Execute training pass on clean data only (Simulating an adversary trying to erase the WM)
    for epoch in range(epochs):
        model.train()
        for inputs, labels in clean_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() #calcolo GRADIENTE --> BACKPROPAGATION
            optimizer.step() #MODIFICA PESI RETE
            
    clean_acc = evaluate_model(model, test_loader, device)
    trig_acc = evaluate_model(model, trigger_loader, device)
    
    del model
    return clean_acc, trig_acc

# =====================================================================
# 3. STOCHASTIC PERTURBATION: GAUSSIAN NOISE INJECTION
# =====================================================================
def run_gaussian_noise_attack(base_model, noise_std, test_loader, trigger_loader, device):
    """
    Injects zero-mean Gaussian noise directly into convolutional and linear layers.
    Simulates physical channel decay or crude network obfuscation.
    """
    model = get_resnet18(pretrained=False).to(device)
    model.load_state_dict(base_model.state_dict())
    model.eval()
    
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Generate noise tensor scaled by the parameter's standard deviation and target intensity
                noise = torch.randn_like(module.weight) * noise_std * module.weight.std()
                module.weight.add_(noise)
                
    clean_acc = evaluate_model(model, test_loader, device)
    trig_acc = evaluate_model(model, trigger_loader, device)
    
    del model
    return clean_acc, trig_acc

# =====================================================================
# 4. STATE-OF-THE-ART ATTACK: FINE-PRUNING
# =====================================================================
def run_fine_pruning_attack(base_model, prune_ratio, clean_train_loader, test_loader, trigger_loader, device, fine_tune_epochs=3):
    """
    Executes Fine-Pruning (Pruning + Fine-Tuning).
    Identifies dormant channels on clean validation data within the final convolutional layer,
    zeroes them out, and performs a brief fine-tuning pass to restore functionality.
    """
    model = get_resnet18(pretrained=False).to(device)
    model.load_state_dict(base_model.state_dict())
    
    # Target the last deep layer where watermarks are statistically localized
    target_layer = model.layer4[1].conv2
    container = {'activations': None}
    
    def forward_hook(module, input, output):
        # Spostiamo subito su CPU per evitare memory leak e accumuli di VRAM su Kaggle
        act = output.detach().mean(dim=(0, 2, 3)).cpu()
        if container['activations'] is None:
            container['activations'] = act.clone()
        else:
            container['activations'] += act.clone()

    hook_handle = target_layer.register_forward_hook(forward_hook)
    
    # Feed a subset of clean training data to profile baseline activations
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(clean_train_loader):
            inputs = inputs.to(device)
            model(inputs)
            if i >= 10: # Profile over 10 batches for stable statistical convergence
                break
    hook_handle.remove()
    
    # Riportiamo il tensore aggregato su device per calcolare il topk
    activations = container['activations'].to(device)
    num_channels = activations.size(0)
    k_prune = int(num_channels * prune_ratio)
    
    _, low_activating_indices = torch.topk(activations, k_prune, largest=False)
    # RIMOSSO .tolist(): manteniamo il tensore su GPU per un indexing ultra-rapido
    
    # Permanently zero out the weights of dormant channels
    with torch.no_grad():
        for idx in low_activating_indices:
            target_layer.weight[idx] = 0.0
            if target_layer.bias is not None:
                target_layer.bias[idx] = 0.0
                
    # Step 3: Post-Pruning Fine-Tuning to repair clean primary classification task
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    
    for epoch in range(fine_tune_epochs):
        model.train()
        for inputs, labels in clean_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # CORREZIONE BUG 2: Azzera i gradienti dei canali potati prima dello step dell'ottimizzatore
            if target_layer.weight.grad is not None:
                target_layer.weight.grad[low_activating_indices] = 0.0
                if target_layer.bias is not None and target_layer.bias.grad is not None:
                    target_layer.bias.grad[low_activating_indices] = 0.0
            
            optimizer.step()
            
            # Enforce hard-masking: assicura che rimangano a zero assoluto dopo lo step
            with torch.no_grad():
                for idx in low_activating_indices:
                    target_layer.weight[idx] = 0.0
                    if target_layer.bias is not None:
                        target_layer.bias[idx] = 0.0

    clean_acc = evaluate_model(model, test_loader, device)
    trig_acc = evaluate_model(model, trigger_loader, device)
    
    del model
    return clean_acc, trig_acc

# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Executing Attack Pipeline on destination device: [{device}]")
    
    # Initialize standard loaders matching reference hyperparameters
    BATCH_SIZE = 128
    clean_train_loader, _, test_loader = get_dataloaders( 
        trigger_dir='../data/trigger_set', 
        total_batch_size=BATCH_SIZE, 
        trigger_size=2
    )
    
    #trasformazione immagini in tensori
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trigger_eval_set = TriggerSetDataset(img_dir='../data/trigger_set', transform=transform_test)
    trigger_eval_loader = DataLoader(trigger_eval_set, batch_size=100, shuffle=False)
    
    # Dictionary storing the historical baseline checkpoints to evaluate
    scenarios = {
        "FROMSCRATCH": "../checkpoints/model_fromscratch.pth",
        "PRETRAINED": "../checkpoints/model_pretrained.pth"
    }
    
    for model_name, path in scenarios.items():
        if not os.path.exists(path):
            print(f"Skipping {model_name} (Checkpoint file not found at {path})")
            continue
            
        print("\n" + "="*60)
        print(f"STRESS-TESTING MODEL PARADIGM: [{model_name}]")
        print("="*60)
        
        # Instantiate skeleton and load optimization state weights
        base_model = get_resnet18(pretrained=False)
        base_model.load_state_dict(torch.load(path, map_location=device))
        base_model = base_model.to(device)
        
        # Calculate initial clean and watermark baseline benchmarks
        init_clean = evaluate_model(base_model, test_loader, device)
        init_trig = evaluate_model(base_model, trigger_eval_loader, device) 
        print(f"Initial Baselines -> Clean Acc: {init_clean:.2f}% | Trigger Acc: {init_trig:.2f}%")
        
        # --- 1. Execute FT / RT Standard Attacks ---
        print("\n--- Running Baseline Paper Attacks (5 Clean Epochs) ---")
        for attack in ["FTLL", "FTAL", "RTLL", "RTAL"]:
            c_acc, t_acc = run_ft_attack(base_model, attack, clean_train_loader, test_loader, trigger_eval_loader, device)
            print(f"[{attack:4}] Post-Attack -> Clean Acc: {c_acc:5.2f}% | Trigger Acc: {t_acc:5.2f}%")
            
        # --- 2. Execute Gaussian Noise Injection Attacks ---
        print("\n--- Running Stochastic Parametric Noise Attacks ---")
        for std in [0.01, 0.05, 0.10]:
            c_acc, t_acc = run_gaussian_noise_attack(base_model, std, test_loader, trigger_eval_loader, device)
            print(f"[Noise Std {std:.2f}] Post-Attack -> Clean Acc: {c_acc:5.2f}% | Trigger Acc: {t_acc:5.2f}%")
            
        # --- 3. Execute Fine-Pruning SOTA Attacks ---
        print("\n--- Running SOTA Fine-Pruning Attacks (Prune + 3 Clean Epochs) ---")
        for ratio in [0.10, 0.30, 0.50]:
            c_acc, t_acc = run_fine_pruning_attack(base_model, ratio, clean_train_loader, test_loader, trigger_eval_loader, device)
            print(f"[Pruned {int(ratio*100)}%] Post-Attack -> Clean Acc: {c_acc:5.2f}% | Trigger Acc: {t_acc:5.2f}%")
            
        # OTTIMIZZAZIONE KAGGLE: Svuota aggressivamente la VRAM prima del prossimo scenario
        del base_model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()