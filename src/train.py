"""
@file train.py
@author Aidan Mohammed-Ali
@brief Master training loop for the Super-Resolution model.
@date 2026-07-28
"""

# =============================
# System Headers
# =============================
import os
import csv

# =============================
# Frameworks
# =============================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data

# =============================
# Custom Modules
# =============================
import dataset
import model

# =============================
# Hyperparameters (Macros)
# =============================
HR_DIR = "data/DIV2K_train_HR"
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# =============================
# Main Execution
# =============================
def main():
    # Hardware detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"[*] Starting training on device: {device}")
    
    # Initialise memory pipeline
    print("[*] Loading Dataset...")
    full_dataset = dataset.SuperResolutionDataset(hr_dir=HR_DIR, crop_size=128, scale_factor=4)
    
    # Calculate sizes for split
    val_size = int(len(full_dataset) * VAL_SPLIT)
    test_size = int(len(full_dataset) * TEST_SPLIT)
    train_size = len(full_dataset) - val_size - test_size
    
    # Split the dataset randomly
    train_dataset, val_dataset, test_dataset = torch_data.random_split(full_dataset, [train_size, val_size, test_size])
    
    # Create separate DataLoaders
    train_loader = torch_data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = torch_data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = torch_data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialise model
    print("[*] Initializing Model...")
    sr_model = model.SuperResolutionModel(scale_factor=4, channels=64, num_residuals=8).to(device)
    
    # Initialise math operators
    criterion = nn.L1Loss()
    optimiser = optim.Adam(sr_model.parameters(), lr=LEARNING_RATE)
    
    # Initialise CSV logger
    csv_file = open("loss_history.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "Train_Loss", "Val_Loss"])
    print("[*] Created loss_history.csv for data logging.")
    
    # Main training loop
    print("[*] Beginning Training Loop...")
    for epoch in range(1, EPOCHS + 1):
        # =============================
        # Training Phase
        # =============================
        sr_model.train()
        epoch_train_loss = 0.0
        
        # Iterate over batches of memory
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Zero out the old calculus gradients from the last loop
            optimiser.zero_grad()
            
            # Forward pass
            fake_hr_imgs = sr_model(lr_imgs)
            
            # Calculate the error between fake and real
            loss = criterion(fake_hr_imgs, hr_imgs)
            
            # Backward pass
            loss.backward()
            
            # Update the model weights using the calculated gradients
            optimiser.step()
            epoch_train_loss += loss.item()
            
            # Output progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        # Calculate average training error for current epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # =============================
        # Validation Phase
        # =============================
        sr_model.eval()
        epoch_val_loss = 0.0
        
        # Disable gradient calculation
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                # Forward pass only
                fake_hr_imgs = sr_model(lr_imgs)
                
                # Calculate validation error
                loss = criterion(fake_hr_imgs, hr_imgs)
                epoch_val_loss += loss.item()
        
        # Calculate average validation error for current epoch
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        print(f"===> Epoch {epoch} Complete. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")
        
        # Write data to CSV and force flush to hard drive immediately
        csv_writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}"])
        csv_file.flush()
        
        # Save a checkpoint of the weights every 10 epochs
        if epoch % 10 == 0:
            save_path = f"checkpoint_epoch_{epoch}.pth"
            torch.save(sr_model.state_dict(), save_path)
            print(f"[*] Saved memory dump to {save_path}")
            
    # Close the file
    csv_file.close()
    print("[*] Training finished successfully. File pointers closed.")
    
    # =============================
    # Testing Phase
    # =============================
    print("\n[*] Beginning Testing Phase...")
    sr_model.eval()
    total_test_loss = 0.0
    
    # Disable gradient calculation
    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Forward pass
            fake_hr_imgs = sr_model(lr_imgs)
            
            # Calculate testing error
            loss = criterion(fake_hr_imgs, hr_imgs)
            total_test_loss += loss.item()
            
    # Calculate average test error
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"[*] ==========================================")
    print(f"[*] Final Test Loss on Unseen Data: {avg_test_loss:.4f}")
    print(f"[*] ==========================================")
    
    # Save the final metric to a text file
    summary_file = open("test_summary.txt", mode="w")
    summary_file.write("==========================================\n")
    summary_file.write("PixelBoost-SR: Final Evaluation Metrics\n")
    summary_file.write("==========================================\n")
    summary_file.write(f"Final Test Loss (L1/MAE): {avg_test_loss:.6f}\n")
    summary_file.close()
    
    print("[*] Saved final test metrics to test_summary.txt")

if __name__ == "__main__":
    main()
