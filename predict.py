from cog import BasePredictor, Input, Path
import cv2
import os
from PIL import Image
import torch
from controlnet_aux import MidasDetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, MLSDdetector

class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load models
        self.depth_detector = MidasDetector.from_pretrained("lllyasviel/Annotators").to(self.device)
        self.openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(self.device)
        self.mask_detector = PidiNetDetector.from_pretrained("lllyasviel/Annotators").to(self.device)
        self.normal_detector = NormalBaeDetector.from_pretrained("lllyasviel/Annotators").to(self.device)
        self.semantic_detector = MLSDdetector.from_pretrained("lllyasviel/Annotators").to(self.device)

        # Ensure output directories exist
        self.output_dirs = ['depth', 'dw_pose', 'mask', 'normal', 'semantic_map']
        for dir_name in self.output_dirs:
            os.makedirs(dir_name, exist_ok=True)

    def process_frame(self, frame, frame_index):
        # Resize frame for model input
        frame_resized = cv2.resize(frame, (319, 236))
        frame_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

        # Perform detections
        depth_image = self.depth_detector(frame_pil)
        pose_image = self.openpose_detector(frame_pil)
        mask_image = self.mask_detector(frame_pil)
        normal_image = self.normal_detector(frame_pil)
        semantic_image = self.semantic_detector(frame_pil)

        # Save processed images
        outputs = {
            'depth': depth_image,
            'dw_pose': pose_image,
            'mask': mask_image,
            'normal': normal_image,
            'semantic_map': semantic_image
        }
        for key, img in outputs.items():
            img.save(f'{key}/{frame_index:04d}_all.png')

    def process_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))  # Ensure path is correctly converted to string
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame, frame_index)
            frame_index += 1
        cap.release()

    def predict(self, video: Path = Input(description="Input video file path")) -> str:
        """Process the video and return the path to the directories containing the outputs."""
        self.process_video(video)
        return "Processed video frames saved in output directories: depth, dw_pose, mask, normal, semantic_map"
