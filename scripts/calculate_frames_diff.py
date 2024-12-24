import os
from PIL import Image, ImageChops
import imageio

def calculate_frames_diff(set_a_path, set_b_path, output_path, gif_name="residual.gif", verbose=True):
    """
    Calculate the residuals between two sets of video frames and save them as images and a GIF.

    Args:
        set_a_path (str): Directory containing the frames of set A.
        set_b_path (str): Directory containing the frames of set B.
        output_path (str): Directory to save the residual images and GIF.
        gif_name (str): Name of the output GIF.
        verbose (bool): Print status messages.
    """
    os.makedirs(output_path, exist_ok=True)

    # Get sorted frame lists
    frames_a = sorted(os.listdir(set_a_path))
    frames_b = sorted(os.listdir(set_b_path))

    assert len(frames_a) == len(frames_b), "Sets A and B must have the same number of frames."

    residual_images = []

    for i, (frame_a_name, frame_b_name) in enumerate(zip(frames_a, frames_b)):
        frame_a_path = os.path.join(set_a_path, frame_a_name)
        frame_b_path = os.path.join(set_b_path, frame_b_name)

        # Open the images
        frame_a = Image.open(frame_a_path).convert("RGB")
        frame_b = Image.open(frame_b_path).convert("RGB")

        # Calculate the residual
        residual = ImageChops.difference(frame_a, frame_b)

        # Save the residual image
        residual_image_path = os.path.join(output_path, f"residual_{i:04d}.png")
        residual.save(residual_image_path)
        residual_images.append(residual)
        if verbose:
            print(f"Saved residual image {i} to {residual_image_path}")

    # Create a GIF
    gif_path = os.path.join(output_path, gif_name)
    residual_images[0].save(
        gif_path,
        save_all=True,
        append_images=residual_images[1:],
        duration=1000 // 8,  # Assuming 8 FPS for the GIF
        loop=0,
    )
    if verbose:
        print(f"Saved residual GIF to {gif_path}")

if __name__ == "__main__":
    org_frames_path = "/home/yfeng/ygcheng/src/Open-Sora/samples/samples/sample_0000_frames"
    ct_frames_path  = "/home/yfeng/ygcheng/src/Open-Sora/samples/ea_samples/sample_0000_ea_0_frames"

    ct_frames_filename = os.path.basename(ct_frames_path)
    output_dir     = "/home/yfeng/ygcheng/src/Open-Sora/outputs/frames_diff"
    output_path    = os.path.join(output_dir, ct_frames_filename)
    calculate_frames_diff(
    set_a_path=org_frames_path,
    set_b_path=ct_frames_path ,
    output_path=output_path,
    gif_name="residual.gif"
)