# app/standard_eval/mnist_eval.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import logging

logger = logging.getLogger(__name__)

# --- Configuration ---
BATCH_SIZE = 128 # Increase batch size if memory allows
NUM_WORKERS = 2 # Adjust based on your system
DATA_ROOT = './data' # Root directory to store dataset

# Ensure dataset directory exists
os.makedirs(DATA_ROOT, exist_ok=True)

def evaluate_network_on_task(model_instance, device):
    """
    Evaluates the PyTorch model on the MNIST test dataset.
    Returns accuracy (0-100) as the fitness score.
    Handles potential dataset download issues and inference errors.

    Args:
        model_instance: The instantiated PyTorch model (nn.Module).
        device: The torch device (e.g., 'cuda' or 'cpu') to run evaluation on.

    Returns:
        float: Accuracy percentage, or -1.0 (or -inf) if evaluation fails significantly.
    """
    logger.info(f"Starting MNIST evaluation for model on device: {device}")
    model_instance.eval() # Set model to evaluation mode
    model_instance.to(device)

    # --- Load MNIST Test Dataset ---
    try:
        logger.info(f"Loading MNIST test dataset from {DATA_ROOT} (downloading if needed)...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific normalization
        ])
        testset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if device.type=='cuda' else False)
        output_classes = 10 # MNIST has 10 classes
        logger.info("MNIST test set loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load or download MNIST dataset: {e}", exc_info=True)
        return -1.0 # Return low fitness score on dataset error

    # --- Evaluation Loop ---
    correct = 0
    total = 0
    total_loss = 0.0
    # Using CrossEntropyLoss is common for classification accuracy tasks
    criterion = torch.nn.CrossEntropyLoss()

    logger.info(f"Starting evaluation loop (Batch Size: {BATCH_SIZE})...")
    evaluation_successful = True
    with torch.no_grad(): # Disable gradient calculation during evaluation
        for i, data in enumerate(testloader):
            try:
                inputs, labels = data
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
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Optional: Log progress periodically
                # if (i + 1) % 50 == 0:
                #    logger.debug(f"  Processed batch {i+1}/{len(testloader)}")


            except Exception as model_err:
                logger.error(f"Error during model inference or loss calculation in batch {i}: {model_err}", exc_info=False) # Avoid flooding logs with tracebacks for every batch error
                # Decide if a single batch error should fail the whole evaluation
                evaluation_successful = False
                break # Stop evaluation on inference error

    # --- Calculate Final Metrics ---
    if not evaluation_successful:
        logger.warning("Evaluation stopped prematurely due to errors.")
        return -1.0 # Return low fitness

    if total == 0:
        logger.warning("Evaluation completed but no samples were processed (total=0).")
        return 0.0 # Or -1.0, depending on how you want to treat this edge case

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(testloader)

    logger.info(f"MNIST Evaluation finished. Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}")

    # --- Return Fitness Score ---
    # Usually accuracy is the fitness for evolution tasks like this
    fitness_score = accuracy
    return fitness_score

