import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# === Orientation Correction ===
def analyze_orientation(frame):
    h, w = frame.shape[:2]
    landscape = w > h
    upside_down = False

    if landscape:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
        ankle_y = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y + landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y) / 2
        if nose_y > ankle_y:
            upside_down = True

    if upside_down:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    return frame

# === Fog Generator ===
def add_fog(frame, intensity=0.2):
    h, w = frame.shape[:2]
    fog_layer = np.random.normal(loc=200, scale=30, size=(h, w)).astype(np.uint8)
    fog_layer = cv2.GaussianBlur(fog_layer, (51, 51), 0)
    fog = np.stack([fog_layer]*3, axis=-1)
    return cv2.addWeighted(frame, 1 - intensity, fog, intensity, 0)

# === Augment Frame ===
def apply_augmentation(frame, config, dx, dy):
    # Rotate
    if config['rotate'] != 0:
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, config['rotate'], 1.0)
        frame = cv2.warpAffine(frame, rot_matrix, (frame.shape[1], frame.shape[0]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Flip (mirror)
    if config['flip']:
        frame = cv2.flip(frame, 1)

    # Translate
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Brightness & Contrast
    frame = cv2.convertScaleAbs(frame, alpha=config['contrast'], beta=config['brightness'])

    # Add Noise
    if config['noise']:
        noise = np.random.normal(0, config['noise_std'], frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add Fog
    if config['fog']:
        frame = add_fog(frame, intensity=config['fog_intensity'])

    return frame

# === Video Augmentation ===
def augment_video(input_path, output_path, config):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {input_path}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print(f"❌ Cannot read first frame: {input_path}")
        cap.release()
        return

    first_frame = analyze_orientation(first_frame)
    height, width = first_frame.shape[:2]
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    dx = int(config['translate_x'] * width)
    dy = int(config['translate_y'] * height)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for _ in tqdm(range(total_frames), desc=os.path.basename(output_path)):
        ret, frame = cap.read()
        if not ret:
            break

        frame = analyze_orientation(frame)
        frame = apply_augmentation(frame, config, dx, dy)
        writer.write(frame)

    cap.release()
    writer.release()

# === Augmentation Config Generator ===
def get_agentic_configs():
    return [
        # Original
        {'flip': False, 'translate_x': 0, 'translate_y': 0, 'brightness': 0, 'contrast': 1.0, 'noise': False, 'noise_std': 0, 'rotate': 0, 'fog': False, 'fog_intensity': 0},

        # Flip only
        {'flip': True, 'translate_x': 0, 'translate_y': 0, 'brightness': 0, 'contrast': 1.0, 'noise': False, 'noise_std': 0, 'rotate': 0, 'fog': False, 'fog_intensity': 0},

        # Translations (7)
        {'flip': False, 'translate_x': -0.10, 'translate_y': 0, 'brightness': 0, 'contrast': 1.0, 'noise': False, 'noise_std': 0, 'rotate': 0, 'fog': True, 'fog_intensity': 0.1},
        {'flip': False, 'translate_x': 0.10, 'translate_y': 0, 'brightness': 0, 'contrast': 1.0, 'noise': False, 'noise_std': 0, 'rotate': 0, 'fog': True, 'fog_intensity': 0.1},
        {'flip': False, 'translate_x': 0, 'translate_y': -0.10, 'brightness': 0, 'contrast': 1.0, 'noise': False, 'noise_std': 0, 'rotate': 0, 'fog': True, 'fog_intensity': 0.1},
        {'flip': False, 'translate_x': 0, 'translate_y': 0.10, 'brightness': 0, 'contrast': 1.0, 'noise': False, 'noise_std': 0, 'rotate': 0, 'fog': True, 'fog_intensity': 0.1},
        {'flip': False, 'translate_x': 0.07, 'translate_y': -0.07, 'brightness': 0, 'contrast': 1.0, 'noise': False, 'noise_std': 0, 'rotate': 0, 'fog': True, 'fog_intensity': 0.1},
        {'flip': False, 'translate_x': -0.07, 'translate_y': 0.07, 'brightness': 0, 'contrast': 1.0, 'noise': False, 'noise_std': 0, 'rotate': 0, 'fog': True, 'fog_intensity': 0.1},
        {'flip': False, 'translate_x': 0.05, 'translate_y': 0.03, 'brightness': 0, 'contrast': 1.0, 'noise': False, 'noise_std': 0, 'rotate': 0, 'fog': True, 'fog_intensity': 0.1},

        # Lighting (5 darkest)
        {'flip': False, 'translate_x': 0, 'translate_y': 0, 'brightness': -25, 'contrast': 0.8, 'noise': False, 'noise_std': 0, 'rotate': 0, 'fog': True, 'fog_intensity': 0.25},
        {'flip': False, 'translate_x': 0, 'translate_y': 0, 'brightness': -30, 'contrast': 0.7, 'noise': False, 'noise_std': 0, 'rotate': 0, 'fog': True, 'fog_intensity': 0.3},
        {'flip': False, 'translate_x': 0, 'translate_y': 0, 'brightness': -35, 'contrast': 0.65, 'noise': False, 'noise_std': 0, 'rotate': 0, 'fog': True, 'fog_intensity': 0.3},
        {'flip': False, 'translate_x': 0, 'translate_y': 0, 'brightness': -20, 'contrast': 0.75, 'noise': False, 'noise_std': 0, 'rotate': 0, 'fog': True, 'fog_intensity': 0.2},
        {'flip': False, 'translate_x': 0, 'translate_y': 0, 'brightness': -15, 'contrast': 0.85, 'noise': False, 'noise_std': 0, 'rotate': 0, 'fog': True, 'fog_intensity': 0.2},

        # Rotations (3)
        {'flip': False, 'translate_x': 0, 'translate_y': 0, 'brightness': 0, 'contrast': 1.0, 'noise': False, 'noise_std': 0, 'rotate': 2, 'fog': True, 'fog_intensity': 0.1},
        {'flip': False, 'translate_x': 0, 'translate_y': 0, 'brightness': 0, 'contrast': 1.0, 'noise': False, 'noise_std': 0, 'rotate': -5, 'fog': True, 'fog_intensity': 0.1},
        {'flip': False, 'translate_x': 0, 'translate_y': 0, 'brightness': 0, 'contrast': 1.0, 'noise': False, 'noise_std': 0, 'rotate': 5, 'fog': True, 'fog_intensity': 0.1},

        # Combined + dark + flip
        {'flip': False, 'translate_x': 0.1, 'translate_y': -0.05, 'brightness': 5, 'contrast': 1.1, 'noise': True, 'noise_std': 4, 'rotate': 2, 'fog': True, 'fog_intensity': 0.2},
        {'flip': True, 'translate_x': 0, 'translate_y': 0, 'brightness': -30, 'contrast': 0.7, 'noise': False, 'noise_std': 0, 'rotate': 0, 'fog': True, 'fog_intensity': 0.25},

        # Moderate noise, no flip
        {'flip': False, 'translate_x': 0, 'translate_y': 0, 'brightness': 0, 'contrast': 1.0, 'noise': True, 'noise_std': 8, 'rotate': 0, 'fog': False, 'fog_intensity': 0},

        # Combined: noise + rotation + fog
        {'flip': False, 'translate_x': -0.05, 'translate_y': 0.05, 'brightness': 0, 'contrast': 1.0, 'noise': True, 'noise_std': 6, 'rotate': -3, 'fog': True, 'fog_intensity': 0.15}

    ]



# === Batch Runner ===
def batch_augment(input_dir, output_dir):
    video_paths = glob.glob(os.path.join(input_dir, '**', '*.mp4'), recursive=True)

    for path in video_paths:
        rel_path = os.path.relpath(path, input_dir)
        base = os.path.splitext(os.path.basename(path))[0]
        subdir = os.path.dirname(rel_path)

        configs = get_agentic_configs()

        for i, config in enumerate(configs):
            outname = f"{base}_aug{i+1:02d}.mp4"
            outpath = os.path.join(output_dir, subdir, outname)
            augment_video(path, outpath, config)

# === Entry Point ===
if __name__ == '__main__':
    INPUT_DIR = "/Users/himanipraneshrao/Desktop/sig/internship/classical/augmentation/data" #change path accordingly
    OUTPUT_DIR = "/Users/himanipraneshrao/Desktop/sig/internship/classical/augmentation/augmented_output"
    batch_augment(INPUT_DIR, OUTPUT_DIR)

#couldn't upload mp3 vidoes due to space restrictions 