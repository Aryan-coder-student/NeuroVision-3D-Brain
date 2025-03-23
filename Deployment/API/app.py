from flask import Flask, request, jsonify
import os
import torch
import nibabel as nib
import numpy as np
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRange,
    CropForeground,
    Resize,
    EnsureType
)

app = Flask(__name__)

class BrainTumorSegmentation:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device , weights_only=True))
        self.model.eval()
        
        self.transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensityRange(
                a_min=-200,
                a_max=200,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            CropForeground(source_key="image"),
            Resize(spatial_size=(128, 128, 128)),
            EnsureType()
        ])

    def get_brats_scan_paths(self, patient_folder):
        """
        Get paths for all modalities from BraTS patient folder
        """
        modality_paths = {
            't1': None,
            't1ce': None,
            't2': None,
            'flair': None
        }
        
        for file in os.listdir(patient_folder):
            if file.endswith('.nii.gz'):
                print(file)
                if 't1.' in file.lower():
                    modality_paths['t1'] = os.path.join(patient_folder, file)
                elif 't1ce.' in file.lower():
                    modality_paths['t1ce'] = os.path.join(patient_folder, file)
                elif 't2.' in file.lower():
                    modality_paths['t2'] = os.path.join(patient_folder, file)
                elif 'flair.' in file.lower():
                    modality_paths['flair'] = os.path.join(patient_folder, file)
        

        missing = [k for k, v in modality_paths.items() if v is None]
        if missing:
            raise ValueError(f"Missing modalities: {missing}")
            
        return [modality_paths['t1'], modality_paths['t1ce'], 
                modality_paths['t2'], modality_paths['flair']]

    def preprocess_scan(self, file_paths):
        """
        Preprocess the 4 modality scans (T1, T1ce, T2, FLAIR)
        """
        processed_scans = []
        for path in file_paths:
            scan = self.transforms(path)
            processed_scans.append(scan)
        
        # Concatenate all modalities
        input_data = torch.cat(processed_scans, dim=0)
        return input_data.unsqueeze(0)  

    @torch.no_grad()
    def predict(self, patient_folder):
        """
        Make prediction on new scans
        patient_folder: Path to patient folder containing all modalities
        """
        # Get paths for all modalities
        file_paths = self.get_brats_scan_paths(patient_folder)
        print("Found all modalities:", file_paths)
        

        input_data = self.preprocess_scan(file_paths).to(self.device)
        
        # Make prediction
        output = self.model(input_data)
        
        # Get segmentation mask
        mask = torch.argmax(output, dim=1)
        return mask.cpu().numpy()[0]  

    def save_prediction(self, mask, output_path, reference_scan):
        """
        Save the prediction mask as a NIfTI file
        """
        # Load reference scan to get affine and header information
        ref_nifti = nib.load(reference_scan)
        
        # Create new NIfTI image with the prediction mask
        pred_nifti = nib.Nifti1Image(mask, ref_nifti.affine, ref_nifti.header)
        nib.save(pred_nifti, output_path)
        print(f"Saved prediction to: {output_path}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        patient_folder = data['patient_folder']
        model_path = "Brain_30.pt"
        
        segmenter = BrainTumorSegmentation(model_path)
        print(f"Processing patient folder: {patient_folder}")
        prediction_mask = segmenter.predict(patient_folder)
        
        output_dir = os.path.join("current_predictions")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "seg.nii.gz")
        reference_scan = segmenter.get_brats_scan_paths(patient_folder)[0] 
        segmenter.save_prediction(prediction_mask, output_path, reference_scan)
        
        return jsonify({"message": "Prediction completed successfully!", "output_path": output_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)