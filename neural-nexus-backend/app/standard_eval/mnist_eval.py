# app/standard_eval/mnist_eval.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F # Keep if loss calculation is used/added
import os
import logging

logger = logging.getLogger(__name__)

# --- Default Configuration (Used if not provided in config dict) ---
DEFAULT_BATCH_SIZE = 128
NUM_WORKERS = 2 # Adjust based on your system
DATA_ROOT = './data' # Root directory to store dataset

# Ensure dataset directory exists
os.makedirs(DATA_ROOT, exist_ok=True)

# --- Renamed function to match expected name ---
# --- Accepts config dict instead of just device ---
def evaluate_model(model_instance: torch.nn.Module, config: dict) -> float:
    """
    Evaluates the PyTorch model on the MNIST test dataset.
    Returns accuracy (0-100) as the fitness score.
    Handles potential dataset download issues and inference errors.

    Args:
        model_instance: The instantiated PyTorch model (nn.Module).
        config (dict): A dictionary containing configuration, expected to have
                       a 'device' key (torch.device or str) and optionally 'batch_size'.

    Returns:
        float: Accuracy percentage, or -float('inf') if evaluation fails significantly.
    """

    # --- Extract Device and Batch Size from Config ---
    device = config.get('device')
    batch_size = config.get('batch_size', DEFAULT_BATCH_SIZE) # Use default if not in config

    # Validate device
    if device is None:
        logger.error("Evaluation config dictionary missing 'device' key.")
        return -float('inf') # Use -inf for consistency with GA fitness expectations
    # Convert device string (e.g., "cuda") to torch.device if needed
    if isinstance(device, str):
        try:
            device = torch.device(device)
        except Exception as e:
             logger.error(f"Invalid device string '{device}' in config: {e}")
             return -float('inf')
    elif not isinstance(device, torch.device):
         logger.error(f"Invalid type for 'device' in config: {type(device)}. Expected torch.device or str.")
         return -float('inf')
    # --- End Device Extraction ---

    logger.info(f"Starting MNIST evaluation for model on device: {device} with Batch Size: {batch_size}")

    # Ensure model is on the correct device and in eval mode
    try:
        model_instance.to(device) # Use the extracted device object
        model_instance.eval() # Set model to evaluation mode
    except Exception as e:
        logger.error(f"Failed to move model to device {device}: {e}", exc_info=True)
        return -float('inf')

    # --- Load MNIST Test Dataset ---
    try:
        logger.info(f"Loading MNIST test dataset from {DATA_ROOT} (downloading if needed)...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific normalization
        ])
        # download=True can cause issues in non-interactive environments if slow/firewalled
        # Consider downloading manually or handling errors more gracefully
        testset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
        # Use batch_size from config
        testloader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True if device.type=='cuda' else False # Use extracted device type
        )
        output_classes = 10 # MNIST has 10 classes
        logger.info("MNIST test set loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load or download MNIST dataset: {e}", exc_info=True)
        return -float('inf') # Return low fitness score on dataset error

    # --- Evaluation Loop ---
    correct = 0
    total = 0
    total_loss = 0.0
    # Using CrossEntropyLoss is standard for classification
    criterion = torch.nn.CrossEntropyLoss(reduction='sum') # Use sum reduction for total loss

    logger.info(f"Starting evaluation loop...")
    evaluation_successful = True
    with torch.no_grad(): # Disable gradient calculation during evaluation
        for i, data in enumerate(testloader):
            try:
                inputs, labels = data
                # Move data to the extracted device
                inputs, labels = inputs.to(device), labels.to(device)

                # --- Model Inference ---
                outputs = model_instance(inputs)

                # --- Validate Output Shape (Optional but recommended) ---
                if outputs.shape[0] != labels.shape[0] or len(outputs.shape) != 2 or outputs.shape[1] != output_classes:
                    logger.error(f"Model output shape mismatch! Expected batch({labels.shape[0]}) x classes({output_classes}), Got: {outputs.shape}. Check model architecture.")
                    evaluation_successful = False
                    break # Stop evaluation if output shape is wrong

                # --- Calculate Loss & Accuracy ---
                loss = criterion(outputs, labels)
                total_loss += loss.item() # .item() gets the scalar value

                _, predicted = torch.max(outputs.data, 1) # Get the index of the max log-probability
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            except Exception as model_err:
                logger.error(f"Error during model inference or loss calculation in batch {i}: {model_err}", exc_info=False) # Avoid flooding logs
                evaluation_successful = False
                break # Stop evaluation on inference error

    # --- Calculate Final Metrics ---
    if not evaluation_successful:
        logger.warning("Evaluation stopped prematurely due to errors.")
        return -float('inf') # Return low fitness

    if total == 0:
        logger.warning("Evaluation completed but no samples were processed (total=0).")
        return 0.0 # Or -float('inf'), depending on desired behavior

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total # Average loss per sample

    logger.info(f"MNIST Evaluation finished. Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}")

    # --- Return Fitness Score ---
    # Using accuracy as fitness
    fitness_score = accuracy
    return float(fitness_score) # Ensure float is returned
