import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Local module imports (assuming all files reside in the same directory)
from data_module import get_dataloaders, get_infinite_iterator, TriggerSetDataset
from models import get_resnet18

def train_epoch(model, clean_loader, criterion, optimizer, device, trigger_iter=None, desc="Training"):
    """
    Trains the model for one epoch, jointly optimizing the primary task 
    and the watermark (if provided). Tracks clean and trigger accuracies separately.
    """
    model.train()
    running_loss = 0.0
    
    # Separate metric tracking to prevent telemetry contamination
    correct_clean, total_clean = 0, 0
    correct_trig, total_trig = 0, 0

    pbar = tqdm(clean_loader, desc=desc, leave=True)

    for clean_inputs, clean_labels in pbar:
        # 1. Safely move clean data to the target device (GPU/CPU)
        clean_inputs = clean_inputs.to(device)
        clean_labels = clean_labels.to(device)
        
        if trigger_iter is not None: #if there is a watermark
            # 2. Extract and move watermark data to the device before concatenation
            trigger_inputs, trigger_labels = next(trigger_iter)
            trigger_inputs = trigger_inputs.to(device)
            trigger_labels = trigger_labels.to(device)
            
            # Concatenate tensors directly on the destination device
            inputs = torch.cat([clean_inputs, trigger_inputs], dim=0)
            labels = torch.cat([clean_labels, trigger_labels], dim=0)
            
            clean_size = clean_labels.size(0)
        else:
            inputs = clean_inputs
            labels = clean_labels

        # Standard optimization pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        
        # 3. Separate metrics computation using tensor slicing
        if trigger_iter is not None:
            # Isolate clean predictions from watermark predictions
            pred_clean = predicted[:clean_size]
            pred_trig = predicted[clean_size:]
            
            total_clean += clean_size
            correct_clean += pred_clean.eq(clean_labels).sum().item()
            
            total_trig += trigger_labels.size(0)
            correct_trig += pred_trig.eq(trigger_labels).sum().item()
            
            acc_clean = 100. * correct_clean / total_clean
            acc_trig = 100. * correct_trig / total_trig
            
            # Update progress bar with decoupled clean and trigger metrics
            pbar.set_postfix(loss=f"{loss.item():.4f}", CleanAcc=f"{acc_clean:.2f}%", TrigAcc=f"{acc_trig:.2f}%")
        else:
            total_clean += labels.size(0)
            correct_clean += predicted.eq(labels).sum().item()
            acc_clean = 100. * correct_clean / total_clean
            pbar.set_postfix(loss=f"{loss.item():.4f}", CleanAcc=f"{acc_clean:.2f}%")

    final_clean_acc = 100. * correct_clean / total_clean
    return running_loss / len(clean_loader), final_clean_acc

def evaluate(model, dataloader, device):
    """
    Evaluates model accuracy on a given dataset.
    Isolates evaluation tasks and skips loss computation for maximum inference efficiency.
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
    return 100. * correct / total

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('../checkpoints', exist_ok=True)

    # Nominal hyperparameters matching the reference paper
    EPOCHS, FT_EPOCHS, BATCH_SIZE = 60, 20, 128
    
    # Initialize Dataloaders (Balanced batch composition constraint with k=2)
    clean_loader, trigger_train_loader, test_loader = get_dataloaders(
        trigger_dir='../data/trigger_set', 
        total_batch_size=BATCH_SIZE, 
        trigger_size=2
    )
    
    # Specific loader for complete Watermark validation (Full Trigger Set)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trigger_eval_set = TriggerSetDataset(img_dir='../data/trigger_set', transform=transform_test)
    trigger_eval_loader = DataLoader(trigger_eval_set, batch_size=100, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------------------
    # Scenario A: Baseline NO-WM (Clean Control Model)
    # -----------------------------------------------------------------
    print("\n--- [1/3] Training NO-WM (Clean Baseline) ---")
    model_nowm = get_resnet18(pretrained=False).to(device)
    optimizer_nowm = optim.SGD(model_nowm.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_nowm = optim.lr_scheduler.StepLR(optimizer_nowm, step_size=20, gamma=0.1)
    
    for epoch in range(EPOCHS):
        train_epoch(model_nowm, clean_loader, criterion, optimizer_nowm, device, desc=f"Epoch {epoch+1}/{EPOCHS}")
        scheduler_nowm.step()
    torch.save(model_nowm.state_dict(), '../checkpoints/model_nowm.pth')

    # -----------------------------------------------------------------
    # Scenario B: FROMSCRATCH (Joint Deep Watermark Injection)
    # -----------------------------------------------------------------
    print("\n--- [2/3] Training FROMSCRATCH (Joint Training) ---")
    model_fs = get_resnet18(pretrained=False).to(device)
    optimizer_fs = optim.SGD(model_fs.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_fs = optim.lr_scheduler.StepLR(optimizer_fs, step_size=20, gamma=0.1)
    trigger_iter = get_infinite_iterator(trigger_train_loader)
    
    for epoch in range(EPOCHS):
        train_epoch(model_fs, clean_loader, criterion, optimizer_fs, device, trigger_iter=trigger_iter, desc=f"Epoch {epoch+1}/{EPOCHS}")
        scheduler_fs.step()
    torch.save(model_fs.state_dict(), '../checkpoints/model_fromscratch.pth')

    # -----------------------------------------------------------------
    # Scenario C: PRETRAINED (Superficial Post-Hoc Fine-Tuning)
    # -----------------------------------------------------------------
    print("\n--- [3/3] Training PRETRAINED (Fine-tuning) ---")
    model_pt = get_resnet18(pretrained=False).to(device)
    model_pt.load_state_dict(torch.load('../checkpoints/model_nowm.pth')) 
    
    optimizer_pt = optim.SGD(model_pt.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    trigger_iter_pt = get_infinite_iterator(trigger_train_loader)
    
    for epoch in range(FT_EPOCHS):
        train_epoch(model_pt, clean_loader, criterion, optimizer_pt, device, trigger_iter=trigger_iter_pt, desc=f"FT Epoch {epoch+1}/{FT_EPOCHS}")
    torch.save(model_pt.state_dict(), '../checkpoints/model_pretrained.pth')

    # -----------------------------------------------------------------
    # Final Summary Table
    # -----------------------------------------------------------------
    print("\n" + "="*30 + "\nFINAL VALIDATION RESULTS\n" + "="*30)
    models_dict = {"NO-WM": model_nowm, "FROMSCRATCH": model_fs, "PRETRAINED": model_pt}
    
    for name, model in models_dict.items():
        clean_acc = evaluate(model, test_loader, device)
        trigger_acc = evaluate(model, trigger_eval_loader, device)
        print(f"[{name:12}] Clean Test Acc: {clean_acc:5.2f}% | Trigger Acc: {trigger_acc:5.2f}%")

if __name__ == '__main__':
    main()