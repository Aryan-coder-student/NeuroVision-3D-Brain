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

    
    stats = calculate_segmentation_statistics(seg_data)

   
    necrotic_core = (seg_data == 1).astype(np.float32)  
    peritumoral_edema = (seg_data == 2).astype(np.float32)  # Peritumoral edema
    enhancing_tumor = (seg_data == 3).astype(np.float32)  # Enhancing tumor

 
    necrotic_mesh = create_3d_mesh(necrotic_core, threshold=0.5)
    edema_mesh = create_3d_mesh(peritumoral_edema, threshold=0.5)
    enhancing_mesh = create_3d_mesh(enhancing_tumor, threshold=0.5)


    t1_mesh = create_3d_mesh(t1_data, threshold=0.2)    # T1
    t1ce_mesh = create_3d_mesh(t1ce_data, threshold=0.2) # T1ce
    t2_mesh = create_3d_mesh(t2_data, threshold=0.2)    # T2
    flair_mesh = create_3d_mesh(flair_data, threshold=0.2) # FLAIR

  
    plotter = pv.Plotter(shape=(2, 2))  # 2x2 grid for multi-planar views

    plotter.set_background("black")

    plotter.subplot(0, 0)
    plotter.add_mesh(necrotic_mesh, color="#FF0000", opacity=0.8, smooth_shading=True, label="Necrotic Tumor Core")  # Red
    plotter.add_mesh(edema_mesh, color="#00FF00", opacity=0.8, smooth_shading=True, label="Peritumoral Edema")      # Green
    plotter.add_mesh(enhancing_mesh, color="#0000FF", opacity=0.8, smooth_shading=True, label="Enhancing Tumor")    # Blue
    plotter.add_mesh(t1_mesh, color="#8B4513", opacity=0.3, smooth_shading=True, label="T1")  # Brown (brain-like)
    plotter.add_legend()
    plotter.add_text("3D Brain View", position="upper_left", font_size=12, color="white")

    # Add axial slice (top-right)
    plotter.subplot(0, 1)
    plotter.add_mesh_slice(t1_mesh, normal=[0, 0, 1], opacity=0.5, name="Axial Slice")
    plotter.add_text("Axial Slice", position="upper_left", font_size=12, color="white")

    # Add sagittal slice (bottom-left)
    plotter.subplot(1, 0)
    plotter.add_mesh_slice(t1_mesh, normal=[1, 0, 0], opacity=0.5, name="Sagittal Slice")
    plotter.add_text("Sagittal Slice", position="upper_left", font_size=12, color="white")

    # Add coronal slice (bottom-right)
    plotter.subplot(1, 1)
    plotter.add_mesh_slice(t1_mesh, normal=[0, 1, 0], opacity=0.5, name="Coronal Slice")
    plotter.add_text("Coronal Slice", position="upper_left", font_size=12, color="white")

    # Add segmentation statistics as text annotations
    stats_text = (
        f"Segmentation Statistics:\n"
        f"Total volume: {stats['total_voxels']} voxels\n"
        f"Necrotic core: {stats['necrotic_voxels']} voxels ({stats['necrotic_percent']:.2f}%)\n"
        f"Peritumoral edema: {stats['edema_voxels']} voxels ({stats['edema_percent']:.2f}%)\n"
        f"Enhancing tumor: {stats['enhancing_voxels']} voxels ({stats['enhancing_percent']:.2f}%)"
    )
    plotter.add_text(
        stats_text,
        position="upper_right",  # Position the text on the right side
        font_size=10,
        color="white",
    )

    # Enable shadows and realistic lighting
    # plotter.enable_shadows()
    # plotter.add_light(pv.Light(position=(1, 1, 1), show_actor=True))

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