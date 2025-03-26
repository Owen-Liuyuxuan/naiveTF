import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from dataset import CacheDataset
from model import PlanningModel

def visualize_prediction(data, predictions, ax, title=None):
    """
    Visualize a single frame with predictions, emphasizing the best mode
    """
    # Plot ego vehicle position (should be at origin)
    ax.plot(0, 0, 'r*', markersize=15, label='Ego Vehicle')
    
    # Plot other vehicles' footprints
    mask = data['objects_mask']
    for i in range(mask.sum()):
        footprint = data['objects_footprint'][i]  # [4, 2]
        velocity = data['objects_velocity'][i]     # [3]
        object_type = data['objects_types'][i]
        
        # Plot footprint
        polygon = Polygon(footprint, fill=True, alpha=0.5, color='lightblue')
        ax.add_patch(polygon)
        
        # Plot velocity vector
        center = footprint.mean(dim=0)
        ax.arrow(center[0].item(), center[1].item(), 
                velocity[0].item(), velocity[1].item(),
                head_width=0.3, head_length=0.5, fc='red', ec='red')
    
    # Plot map elements
    for i in range(data['boundary_mask'][..., 0].sum()):
        left = data['boundary_left_boundaries'][i]   # [lane_point_number, 2]
        right = data['boundary_right_boundaries'][i] # [lane_point_number, 2]

        if data["boundary_in_route"][i]:
            ax.plot(left[:, 0], left[:, 1], 'k-', linewidth=2, alpha=0.8)
            ax.plot(right[:, 0], right[:, 1], 'k-', linewidth=2, alpha=0.8)
        else:
            ax.plot(left[:, 0], left[:, 1], 'k-', linewidth=1, alpha=0.3)
            ax.plot(right[:, 0], right[:, 1], 'k-', linewidth=1, alpha=0.3)
    
    # Plot ground truth trajectory
    history_positions = data['history_trajectories_transform'][:, :2, 3]  # [N_h, 2]
    future_positions = data['future_trajectories_transform'][:, :2, 3]    # [N_f, 2]
    
    ax.plot(history_positions[:, 0], history_positions[:, 1], 'b-', 
            linewidth=2, label='History', alpha=0.7)
    ax.plot(future_positions[:, 0], future_positions[:, 1], 'g-', 
            linewidth=2, label='Ground Truth', alpha=0.7)
    
    # Plot predicted trajectories
    probabilities = predictions['probability'][0]  # [num_modes]
    best_mode = probabilities.argmax().item()
    
    # Create color map for different modes
    num_modes = predictions['trajectory'].shape[1]
    base_colors = plt.cm.rainbow(np.linspace(0, 1, num_modes))
    
    # Plot non-best modes first (more transparent)
    for mode in range(num_modes):
        if mode == best_mode:
            continue
            
        traj = predictions['trajectory'][0, mode, :, :2]  # [steps, 2]
        prob = probabilities[mode].item()
        
        # Scale alpha by probability relative to best probability
        relative_prob = prob / probabilities[best_mode].item()
        alpha = max(0.1, relative_prob * 0.5)  # minimum alpha of 0.1
        
        color = list(base_colors[mode])
        color[-1] = alpha  # Set alpha value
        
        ax.plot(traj[:, 0], traj[:, 1], '--', color=color, 
                label=f'Mode {mode} ({prob:.2f})', linewidth=1)
    
    # Plot best mode last (most visible)
    best_traj = predictions['trajectory'][0, best_mode, :, :2]
    best_prob = probabilities[best_mode].item()
    
    ax.plot(best_traj[:, 0], best_traj[:, 1], '-', color=base_colors[best_mode], 
            label=f'Best Mode {best_mode} ({best_prob:.2f})', 
            linewidth=3, alpha=0.8)
    
    # Set equal aspect ratio and add labels
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    if title:
        ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set reasonable view limits
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    
    # Add grid
    ax.grid(True, alpha=0.3)

import pygame
import sys
from pygame import gfxdraw
import math

def pygame_visualization_mode(dataset, model, device, window_size=(1000, 1000)):
    """
    Interactive Pygame visualization mode for the entire dataset
    """
    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Trajectory Prediction Visualization")
    clock = pygame.time.Clock()
    
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    GRAY = (128, 128, 128)
    
    def world_to_screen(point, scale=10, offset=(500, 500)):
        """Convert world coordinates to screen coordinates"""
        x = int(point[0] * scale + offset[0])
        y = int(-point[1] * scale + offset[1])  # Flip y-axis
        return (x, y)
    
    def draw_polygon(surface, points, color, alpha=255):
        """Draw a polygon with alpha transparency"""
        if len(points) < 3:
            return
        
        points = [(int(x), int(y)) for x, y in points]
        
        # Draw filled polygon with alpha
        gfxdraw.filled_polygon(surface, points, (*color, alpha))
        # Draw polygon outline
        gfxdraw.aapolygon(surface, points, (*color, alpha))
    
    def draw_arrow(surface, start, end, color, width=2):
        """Draw an arrow from start to end"""
        pygame.draw.line(surface, color, start, end, width)
        
        # Calculate arrow head
        angle = math.atan2(start[1]-end[1], end[0]-start[0])
        arrow_size = 10
        arrow_angle = math.pi/6
        
        point1 = (end[0] - arrow_size * math.cos(angle - arrow_angle),
                 end[1] + arrow_size * math.sin(angle - arrow_angle))
        point2 = (end[0] - arrow_size * math.cos(angle + arrow_angle),
                 end[1] + arrow_size * math.sin(angle + arrow_angle))
        
        pygame.draw.polygon(surface, color, [end, point1, point2])
    
    current_idx = 0
    scale = 10  # Scale factor for visualization
    offset = (window_size[0]//2, window_size[1]//2)  # Center offset
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    current_idx = (current_idx + 1) % len(dataset)
                elif event.key == pygame.K_LEFT:
                    current_idx = (current_idx - 1) % len(dataset)
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Clear screen
        screen.fill(WHITE)
        
        # Get current data sample
        data = dataset[current_idx]
        data_batch = {k: v.unsqueeze(0).contiguous().to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in data.items()}
        
        # Get model predictions
        with torch.no_grad():
            predictions = model(data_batch)
        
        # Draw map elements
        for i in range(data['boundary_mask'][..., 0].sum()):
            left = data['boundary_left_boundaries'][i].cpu().numpy()
            right = data['boundary_right_boundaries'][i].cpu().numpy()
            
            # Convert points to screen coordinates
            left_screen = [world_to_screen(p, scale, offset) for p in left]
            right_screen = [world_to_screen(p, scale, offset) for p in right]
            
            color = BLACK if data["boundary_in_route"][i] else GRAY
            width = 2 if data["boundary_in_route"][i] else 1
            alpha = 200 if data["boundary_in_route"][i] else 100
            
            # Draw road boundaries
            if len(left_screen) > 1:
                pygame.draw.lines(screen, color, False, left_screen, width)
            if len(right_screen) > 1:
                pygame.draw.lines(screen, color, False, right_screen, width)
        
        # Draw other vehicles
        mask = data['objects_mask']
        for i in range(mask.sum()):
            footprint = data['objects_footprint'][i].cpu().numpy()
            velocity = data['objects_velocity'][i].cpu().numpy()
            
            # Convert footprint to screen coordinates
            footprint_screen = [world_to_screen(p[:2], scale, offset) for p in footprint]
            
            # Draw vehicle footprint
            draw_polygon(screen, footprint_screen, (100, 100, 255), alpha=128)
            
            # Draw velocity vector
            center = np.mean(footprint, axis=0)
            center_screen = world_to_screen(center[:2], scale, offset)
            vel_end = world_to_screen(center[:2] + velocity[:2], scale, offset)
            draw_arrow(screen, center_screen, vel_end, RED)
        
        # Draw trajectories
        # History
        history = data['history_trajectories_transform'][:, :2, 3].cpu().numpy()
        history_screen = [world_to_screen(p, scale, offset) for p in history]
        if len(history_screen) > 1:
            pygame.draw.lines(screen, BLUE, False, history_screen, 2)
        
        # # Ground truth future
        # future = data['future_trajectories_transform'][:, :2, 3].cpu().numpy()
        # future_screen = [world_to_screen(p, scale, offset) for p in future]
        # if len(future_screen) > 1:
        #     pygame.draw.lines(screen, GREEN, False, future_screen, 2)
        
        # Predicted trajectories
        probabilities = predictions['probability'][0].cpu()
        best_mode = probabilities.argmax().item()
        
        for mode in range(predictions['trajectory'].shape[1]):
            traj = predictions['trajectory'][0, mode, :, :2].cpu().numpy()
            prob = probabilities[mode].item()
            
            # Convert trajectory to screen coordinates
            traj_screen = [world_to_screen(p, scale, offset) for p in traj]
            
            if mode == best_mode:
                color = (255, 165, 0)  # Orange for best mode
                width = 4
                alpha = 255
            else:
                color = (220, 220, 220)
                width = 1
                alpha = int(128 * (prob / probabilities[best_mode].item()))
            
            if len(traj_screen) > 1:
                pygame.draw.lines(screen, color, False, traj_screen, width)
        
        # Draw ego vehicle
        pygame.draw.circle(screen, RED, world_to_screen((0, 0), scale, offset), 5)
        
        # Add text information
        font = pygame.font.Font(None, 36)
        text = font.render(f"Frame {data['frame_id']} ({current_idx}/{len(dataset)})", True, BLACK)
        screen.blit(text, (10, 10))
        
        # Update display
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='Test and Visualize Planning Transformer')
    parser.add_argument('--cache_file', type=str, required=True, help='Path to cache file')
    parser.add_argument('--map_file', type=str, required=True, help='Path to map file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--mode', type=str, choices=['static', 'interactive'], default='static',
                      help='Visualization mode: static (save images) or interactive (pygame)')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize in static mode')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='visualization_outputs')
    args = parser.parse_args()
    
    # Initialize dataset
    dataset = CacheDataset(args.cache_file, args.map_file)
    
    # Initialize model and load weights
    model = PlanningModel().to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if args.mode == 'interactive':
        pygame_visualization_mode(dataset, model, args.device)
    else:
        # Original static visualization code
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        indices = np.random.choice(len(dataset), args.num_samples, replace=False)
        for idx in indices:
            data = dataset[idx]
            # Add batch dimension and move to device
            data = {k: v.unsqueeze(0).contiguous().to(args.device) if isinstance(v, torch.Tensor) else v 
                for k, v in data.items()}
            
            # Get predictions
            with torch.no_grad():
                predictions = model(data)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # Visualize
            visualize_prediction(
                {k: v[0].cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()},
                {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in predictions.items()},
                ax,
                title=f"Frame {data['frame_id']} - Best ADE: {checkpoint['best_ade']:.4f}"
            )
            
            # Save figure
            plt.savefig(output_dir / f'prediction_{idx}.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Saved visualization for frame {data['frame_id']} to {output_dir}/prediction_{idx}.png")

if __name__ == '__main__':
    main()