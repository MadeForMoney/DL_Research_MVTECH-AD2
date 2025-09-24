# import cv2
# import numpy as np
# import os # Import the os module

# def refine_mask(mask: np.ndarray, close_kernel_size: int = 7, open_kernel_size: int = 7) -> np.ndarray:
#     """
#     Refines a binary anomaly mask using a sequence of morphological operations.

#     This function performs:
#     1.  A morphological closing to fill small holes within anomaly regions and connect
#         nearby components.
#     2.  A morphological opening to remove small, isolated noise pixels (false positives).

#     Args:
#         mask (np.ndarray): The input raw binary mask (should be a 2D array with
#                            values of 0 and 255).
#         close_kernel_size (int): The size of the kernel for the closing operation.
#                                  Larger values fill larger holes.
#         open_kernel_size (int): The size of the kernel for the opening operation.
#                                 Larger values remove larger noise specks.

#     Returns:
#         np.ndarray: The refined binary mask.
#     """
#     # Ensure the input mask is a binary 2D array of type uint8
#     if mask.ndim > 2:
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

#     # Binarize the mask to be sure it's 0 or 255
#     _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

#     # --- 1. Morphological Closing ---
#     close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
#     closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel)

#     # --- 2. Morphological Opening ---
#     open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
#     opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, open_kernel)
    
#     return opened_mask

# # --- Example Usage ---
# if __name__ == '__main__':
#     # This block will run if you execute this script directly
    
#     # --- FILLED IN: Create the output directory if it doesn't exist ---
#     output_dir = "results"
#     os.makedirs(output_dir, exist_ok=True)

#     # Create a dummy mask with noise and a hole for demonstration
#     dummy_mask = np.zeros((200, 200), dtype=np.uint8)
#     # A large region with a hole in the middle
#     cv2.rectangle(dummy_mask, (50, 50), (150, 150), 255, -1)
#     cv2.rectangle(dummy_mask, (90, 90), (110, 110), 0, -1)
#     # Some salt noise (white specks)
#     dummy_mask[20:25, 20:25] = 255
#     dummy_mask[170:175, 170:175] = 255

#     # Apply the refinement
#     refined_dummy_mask = refine_mask(dummy_mask)

#     # Save the results to inspect them
#     before_path = os.path.join(output_dir, "dummy_mask_before.png")
#     after_path = os.path.join(output_dir, "dummy_mask_after.png")
#     cv2.imwrite(before_path, dummy_mask)
#     cv2.imwrite(after_path, refined_dummy_mask)

#     print("Dummy mask refinement complete.")
#     print(f"Check '{before_path}' and '{after_path}'.")


import cv2
import numpy as np
import os

def postprocess_anomaly_map(anomaly_map: np.ndarray, threshold_pct: float = 0.5, close_kernel_size: int = 7, open_kernel_size: int = 7) -> np.ndarray:
    """
    Post-processes a raw, grayscale anomaly map to produce a clean, binary mask.

    This function performs:
    1.  A binary thresholding to create a preliminary mask.
    2.  A morphological closing to fill holes and connect nearby regions.
    3.  A morphological opening to remove small noise pixels.

    Args:
        anomaly_map (np.ndarray): The raw (H, W) grayscale anomaly map from the model.
        threshold_pct (float): The percentile threshold to binarize the map (e.g., 0.5 means
                                 the top 50% of anomalous scores become white).
        close_kernel_size (int): The kernel size for the closing operation.
        open_kernel_size (int): The kernel size for the opening operation.

    Returns:
        np.ndarray: The final, cleaned binary anomaly mask (0 or 255).
    """
    # 1. Thresholding
    # Find the threshold value at the given percentile of the map's intensities.
    threshold_value = np.percentile(anomaly_map, 100 * (1 - threshold_pct))
    _, binary_mask = cv2.threshold(anomaly_map, threshold_value, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)

    # 2. Morphological Closing
    # Fills small holes in the detected anomaly regions.
    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel)

    # 3. Morphological Opening
    # Removes small, isolated "salt" noise from the mask.
    open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, open_kernel)
    
    return opened_mask

# --- Example Usage ---
# if __name__ == '__main__':
#     output_dir = "results"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Create a dummy grayscale anomaly map for demonstration
#     # It has a main region with a hole, some noise, and varying intensities.
#     dummy_anomaly_map = np.zeros((200, 200), dtype=np.uint8)
#     # A large region with a hole in the middle
#     cv2.rectangle(dummy_anomaly_map, (50, 50), (150, 150), 180, -1) # Main region
#     cv2.rectangle(dummy_anomaly_map, (90, 90), (110, 110), 0, -1)     # Hole
#     # Some salt noise (isolated bright specks)
#     cv2.rectangle(dummy_anomaly_map, (20, 20), (25, 25), 250, -1)   
#     cv2.rectangle(dummy_anomaly_map, (170, 170), (175, 175), 220, -1)
    
#     # Apply the post-processing
#     # We'll use a threshold that keeps the top 30% of scores.
#     refined_mask = postprocess_anomaly_map(dummy_anomaly_map, threshold_pct=0.3)

#     # Save the results to inspect them
#     before_path = os.path.join(output_dir, "anomaly_map_before.png")
#     after_path = os.path.join(output_dir, "anomaly_map_after.png")
    
#     # Apply a colormap for better visualization of the "before" state
#     anomaly_map_color = cv2.applyColorMap(dummy_anomaly_map, cv2.COLORMAP_INFERNO)
    
#     cv2.imwrite(before_path, anomaly_map_color)
#     cv2.imwrite(after_path, refined_mask)

#     print("Anomaly map post-processing complete.")
#     print(f"Check '{before_path}' (heatmap) and '{after_path}' (binary mask).")
