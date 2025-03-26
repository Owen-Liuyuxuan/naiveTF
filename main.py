import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


from dataset import CacheDataset
from model import PlanningModel

def compute_trajectory_error(pred_trajectory, gt_trajectory):
    """
    Compute trajectory prediction error.
    
    Args:
        pred_trajectory: [B, num_modes, steps, 4] (x,y,dx,dy)
        gt_trajectory: [B, steps, 4, 4] (transformation matrices)
    
    Returns:
        dict containing various error metrics
    """
    # Extract positions from ground truth transformation matrices
    gt_positions = gt_trajectory[..., :2, 3]  # [B, steps, 2]
    
    # Compute ADE (Average Displacement Error) for each mode
    displacement_error = torch.norm(
        pred_trajectory[..., :2] - gt_positions.unsqueeze(1), 
        dim=-1
    )  # [B, num_modes, steps]
    
    ade = displacement_error.mean(dim=-1)  # [B, num_modes]
    
    # Compute FDE (Final Displacement Error) for each mode
    fde = displacement_error[..., -1]  # [B, num_modes]
    
    # Get best mode based on ADE
    best_mode_indices = ade.argmin(dim=-1)  # [B]
    min_ade = ade.min(dim=-1)[0]  # [B]
    min_fde = torch.gather(fde, 1, best_mode_indices.unsqueeze(-1)).squeeze(-1)  # [B]
    
    return {
        'ade': ade,
        'fde': fde,
        'min_ade': min_ade,
        'min_fde': min_fde,
        'best_mode_indices': best_mode_indices
    }

def compute_loss(pred, target, reduction='mean'):
    """
    Compute training loss combining trajectory error and mode probability.
    
    Args:
        pred: dict containing 'trajectory' and 'probability'
        target: ground truth future trajectories
        reduction: how to reduce the loss
    """
    pred_trajectory = pred['trajectory']  # [B, num_modes, steps, 4]
    pred_probability = pred['probability']  # [B, num_modes]
    
    # Compute trajectory errors
    error_dict = compute_trajectory_error(pred_trajectory, target)
    
    # Compute negative log likelihood loss for the best mode
    batch_indices = torch.arange(pred_probability.shape[0], device=pred_probability.device)
    best_mode_indices = error_dict['best_mode_indices']
    best_mode_probs = pred_probability[batch_indices, best_mode_indices]
    prob_loss = -torch.log(best_mode_probs + 1e-6)
    
    # Combine trajectory and probability losses
    trajectory_loss = error_dict['min_ade']
    
    total_loss = trajectory_loss + 0.5 * prob_loss
    
    if reduction == 'mean':
        return total_loss.mean()
    elif reduction == 'none':
        return total_loss
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        # Move data to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch)
        
        # Compute loss
        loss = compute_loss(predictions, batch['future_trajectories_transform'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches

def evaluate(model, dataloader, device):
    model.eval()
    total_ade = 0
    total_fde = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            predictions = model(batch)
            error_dict = compute_trajectory_error(
                predictions['trajectory'], 
                batch['future_trajectories_transform']
            )
            
            total_ade += error_dict['min_ade'].sum().item()
            total_fde += error_dict['min_fde'].sum().item()
            num_samples += batch['future_trajectories_transform'].shape[0]
    
    return {
        'average_ade': total_ade / num_samples,
        'average_fde': total_fde / num_samples
    }

def main():
    parser = argparse.ArgumentParser(description='Train Planning Transformer')
    parser.add_argument('--train_cache', type=str, required=True, 
                       help='Path to training cache file(s) or directory')
    parser.add_argument('--val_cache', type=str, required=True,
                       help='Path to validation cache file(s) or directory')
    parser.add_argument('--map_file', type=str, required=True, 
                       help='Path to map file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(output_dir / 'logs')
    
    # Initialize datasets and dataloaders
    train_dataset = CacheDataset(args.train_cache, args.map_file)
    val_dataset = CacheDataset(args.val_cache, args.map_file)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = PlanningModel().to(args.device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    best_ade = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, args.device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Evaluate
        eval_metrics = evaluate(model, val_dataloader, args.device)
        writer.add_scalar('ADE/val', eval_metrics['average_ade'], epoch)
        writer.add_scalar('FDE/val', eval_metrics['average_fde'], epoch)
        
        # Save best model
        if eval_metrics['average_ade'] < best_ade:
            best_ade = eval_metrics['average_ade']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ade': best_ade,
            }, output_dir / 'best_model.pth')
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val ADE: {eval_metrics['average_ade']:.4f}")
        print(f"Val FDE: {eval_metrics['average_fde']:.4f}")
    
    writer.close()

if __name__ == '__main__':
    main()