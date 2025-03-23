import os
import nibabel as nib
import numpy as np
import pyvista as pv

def load_nifti(file_path):
    """Load a NIfTI file and return its data and affine matrix."""
    nifti = nib.load(file_path)
    data = nifti.get_fdata()
    affine = nifti.affine
    return data, affine

def create_3d_mesh(data, threshold=0.5):
    """Create a 3D mesh from the segmentation data using a threshold."""
    grid = pv.ImageData()
    grid.dimensions = data.shape
    grid.spacing = (1, 1, 1)  # Adjust spacing if needed
    grid.point_data["values"] = data.flatten(order="F")
    mesh = grid.contour([threshold])
    return mesh

def calculate_segmentation_statistics(seg_data):
    """Calculate segmentation statistics."""
    total_voxels = seg_data.size
    necrotic_voxels = np.sum(seg_data == 1)
    edema_voxels = np.sum(seg_data == 2)
    enhancing_voxels = np.sum(seg_data == 3)

    necrotic_percent = (necrotic_voxels / total_voxels) * 100
    edema_percent = (edema_voxels / total_voxels) * 100
    enhancing_percent = (enhancing_voxels / total_voxels) * 100

    return {
        "total_voxels": total_voxels,
        "necrotic_voxels": necrotic_voxels,
        "edema_voxels": edema_voxels,
        "enhancing_voxels": enhancing_voxels,
        "necrotic_percent": necrotic_percent,
        "edema_percent": edema_percent,
        "enhancing_percent": enhancing_percent,
    }

def visualize_3d_brain(segmentation_path, t1_path, t1ce_path, t2_path, flair_path):
    """Visualize the 3D brain with segmentation and other modalities."""
    # Load the segmentation mask
    seg_data, seg_affine = load_nifti(segmentation_path)
    
    # Load other modalities
    t1_data, _ = load_nifti(t1_path)
    t1ce_data, _ = load_nifti(t1ce_path)
    t2_data, _ = load_nifti(t2_path)
    flair_data, _ = load_nifti(flair_path)

    # Calculate segmentation statistics
    stats = calculate_segmentation_statistics(seg_data)

    # Separate the segmentation into three parts
    necrotic_core = (seg_data == 1).astype(np.float32)  # Necrotic tumor core
    peritumoral_edema = (seg_data == 2).astype(np.float32)  # Peritumoral edema
    enhancing_tumor = (seg_data == 3).astype(np.float32)  # Enhancing tumor

    # Create meshes for each segmentation part
    necrotic_mesh = create_3d_mesh(necrotic_core, threshold=0.5)
    edema_mesh = create_3d_mesh(peritumoral_edema, threshold=0.5)
    enhancing_mesh = create_3d_mesh(enhancing_tumor, threshold=0.5)

    # Create meshes for other modalities
    t1_mesh = create_3d_mesh(t1_data, threshold=0.2)    # T1
    t1ce_mesh = create_3d_mesh(t1ce_data, threshold=0.2) # T1ce
    t2_mesh = create_3d_mesh(t2_data, threshold=0.2)    # T2
    flair_mesh = create_3d_mesh(flair_data, threshold=0.2) # FLAIR

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Set background color to black
    plotter.set_background("black")

    # Add segmentation meshes to the plotter with custom colors
    plotter.add_mesh(necrotic_mesh, color="#FF0000", opacity=0.8, label="Necrotic Tumor Core")  # Red
    plotter.add_mesh(edema_mesh, color="#00FF00", opacity=0.8, label="Peritumoral Edema")      # Green
    plotter.add_mesh(enhancing_mesh, color="#0000FF", opacity=0.8, label="Enhancing Tumor")    # Blue

    # Add other modality meshes to the plotter with a brain-like color theme
    plotter.add_mesh(t1_mesh, color="#8B4513", opacity=0.3, label="T1")  # Brown (brain-like)
    plotter.add_mesh(t1ce_mesh, color="#8B4513", opacity=0.3, label="T1ce")  # Brown
    plotter.add_mesh(t2_mesh, color="#8B4513", opacity=0.3, label="T2")  # Brown
    plotter.add_mesh(flair_mesh, color="#8B4513", opacity=0.3, label="FLAIR")  # Brown

    # Add a legend
    plotter.add_legend()

    # Add segmentation statistics as text annotations on the left side
    stats_text = (
        f"Segmentation Statistics:\n"
        f"Total volume: {stats['total_voxels']} voxels\n"
        f"Necrotic core: {stats['necrotic_voxels']} voxels ({stats['necrotic_percent']:.2f}%)\n"
        f"Peritumoral edema: {stats['edema_voxels']} voxels ({stats['edema_percent']:.2f}%)\n"
        f"Enhancing tumor: {stats['enhancing_voxels']} voxels ({stats['enhancing_percent']:.2f}%)"
    )
    plotter.add_text(
        stats_text,
        position="upper_left",  # Position the text on the left side
        font_size=12,
        color="white",
    )

    # Set up the camera and show the plot
    plotter.show()

if __name__ == "__main__":
    # Paths to the NIfTI files
    segmentation_path = "current_predictions/seg.nii.gz"  # Predicted segmentation
    t1_path = "BraTS2021_00000/Paitent 1/t1.nii.gz"      # T1 modality
    t1ce_path = "BraTS2021_00000/Paitent 1/t1ce.nii.gz"  # T1ce modality
    t2_path = "BraTS2021_00000/Paitent 1/t2.nii.gz"      # T2 modality
    flair_path = "BraTS2021_00000/Paitent 1/flair.nii.gz" # FLAIR modality

    # Visualize the 3D brain
    visualize_3d_brain(segmentation_path, t1_path, t1ce_path, t2_path, flair_path)