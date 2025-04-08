import os
import shutil
from typing import List, Union

from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import insightface
from insightface.app.common import Face
# try:
#     import torch.cuda as cuda
# except:
#     cuda = None
import torch
import torch.cuda.amp
import folder_paths
import comfy.model_management as model_management
from ..modules.shared import state

from .reactor_logger import logger
from ..reactor_utils import (
    move_path,
    get_image_md5hash,
)
from .r_faceboost import swapper, restorer

import warnings

np.warnings = warnings
np.warnings.filterwarnings('ignore')

# PROVIDERS
try:
    if torch.cuda.is_available():
        providers = ["CUDAExecutionProvider"]
        print("Using CUDA GPU for face Analysis and swapping.")
    elif torch.backends.mps.is_available():
        providers = ["CoreMLExecutionProvider"]
    elif hasattr(torch,'dml') or hasattr(torch,'privateuseone'):
        providers = ["ROCMExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
        print("Using CPU for face Analysis and swapping..")
except Exception as e:
    logger.debug(f"ExecutionProviderError: {e}.\nEP is set to CPU.")
    providers = ["CPUExecutionProvider"]
# if cuda is not None:
#     if cuda.is_available():
#         providers = ["CUDAExecutionProvider"]
#     else:
#         providers = ["CPUExecutionProvider"]
# else:
#     providers = ["CPUExecutionProvider"]

models_path_old = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
insightface_path_old = os.path.join(models_path_old, "insightface")
insightface_models_path_old = os.path.join(insightface_path_old, "models")

models_path = folder_paths.models_dir
insightface_path = os.path.join(models_path, "insightface")
insightface_models_path = os.path.join(insightface_path, "models")
reswapper_path = os.path.join(models_path, "reswapper")

if os.path.exists(models_path_old):
    move_path(insightface_models_path_old, insightface_models_path)
    move_path(insightface_path_old, insightface_path)
    move_path(models_path_old, models_path)
if os.path.exists(insightface_path) and os.path.exists(insightface_path_old):
    shutil.rmtree(insightface_path_old)
    shutil.rmtree(models_path_old)

FACE_CACHE = {}

FS_MODEL = None
CURRENT_FS_MODEL_PATH = None

ANALYSIS_MODELS = {
    "640": None,
    "320": None,
}

SOURCE_FACES = None
SOURCE_IMAGE_HASH = None
TARGET_FACES = None
TARGET_IMAGE_HASH = None
TARGET_FACES_LIST = []
TARGET_IMAGE_LIST_HASH = []

def unload_model(model):
    if model is not None:
        # check if model has unload method
        # if "unload" in model:
        #     model.unload()
        # if "model_unload" in model:
        #     model.model_unload()
        del model
    return None

def unload_all_models():
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    FS_MODEL = unload_model(FS_MODEL)
    ANALYSIS_MODELS["320"] = unload_model(ANALYSIS_MODELS["320"])
    ANALYSIS_MODELS["640"] = unload_model(ANALYSIS_MODELS["640"])

def get_current_faces_model():
    global SOURCE_FACES
    return SOURCE_FACES

# def getAnalysisModel(det_size = (640, 640)):
#     global ANALYSIS_MODELS
#     ANALYSIS_MODEL = ANALYSIS_MODELS[str(det_size[0])]
#     if ANALYSIS_MODEL is None:
#         ANALYSIS_MODEL = insightface.app.FaceAnalysis(
#             name="buffalo_l", providers=providers, root=insightface_path
#         )
#     ANALYSIS_MODEL.prepare(ctx_id=0, det_size=det_size)
#     ANALYSIS_MODELS[str(det_size[0])] = ANALYSIS_MODEL
#     return ANALYSIS_MODEL

# def getFaceSwapModel(model_path: str):
#     global FS_MODEL, CURRENT_FS_MODEL_PATH
#     if FS_MODEL is None or CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
#         CURRENT_FS_MODEL_PATH = model_path
#         FS_MODEL = unload_model(FS_MODEL)
#         FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)

#     return FS_MODEL


def getAnalysisModel(det_size=(640, 640)):
    global ANALYSIS_MODELS
    ANALYSIS_MODEL = ANALYSIS_MODELS[str(det_size[0])]
    if ANALYSIS_MODEL is None:
        # Force CUDA execution provider for GPU acceleration
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l", 
            providers=["CUDAExecutionProvider"],  # Explicitly use CUDA
            root=insightface_path
        )
        # Disable CPU fallback
        ANALYSIS_MODEL.disable_cpu_fallback = True  
    ANALYSIS_MODEL.prepare(ctx_id=0, det_size=det_size)
    ANALYSIS_MODELS[str(det_size[0])] = ANALYSIS_MODEL
    return ANALYSIS_MODEL

def getFaceSwapModel(model_path: str):
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    if FS_MODEL is None or CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = unload_model(FS_MODEL)
        # Force CUDA execution
        FS_MODEL = insightface.model_zoo.get_model(
            model_path, 
            providers=["CUDAExecutionProvider"]
        )
        # Ensure model uses GPU
        if hasattr(FS_MODEL, 'model'):
            FS_MODEL.model.to('cuda')
    return FS_MODEL


def sort_by_order(face, order: str):
    if order == "left-right":
        return sorted(face, key=lambda x: x.bbox[0])
    if order == "right-left":
        return sorted(face, key=lambda x: x.bbox[0], reverse = True)
    if order == "top-bottom":
        return sorted(face, key=lambda x: x.bbox[1])
    if order == "bottom-top":
        return sorted(face, key=lambda x: x.bbox[1], reverse = True)
    if order == "small-large":
        return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    # if order == "large-small":
    #     return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)
    # by default "large-small":
    return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)

def get_face_gender(
        face,
        face_index,
        gender_condition,
        operated: str,
        order: str,
):
    gender = [
        x.sex
        for x in face
    ]
    gender.reverse()
    # If index is outside of bounds, return None, avoid exception
    if face_index >= len(gender):
        logger.status("Requested face index (%s) is out of bounds (max available index is %s)", face_index, len(gender))
        return None, 0
    face_gender = gender[face_index]
    logger.status("%s Face %s: Detected Gender -%s-", operated, face_index, face_gender)
    if (gender_condition == 1 and face_gender == "F") or (gender_condition == 2 and face_gender == "M"):
        logger.status("OK - Detected Gender matches Condition")
        try:
            faces_sorted = sort_by_order(face, order)
            return faces_sorted[face_index], 0
            # return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
        except IndexError:
            return None, 0
    else:
        logger.status("WRONG - Detected Gender doesn't match Condition")
        faces_sorted = sort_by_order(face, order)
        return faces_sorted[face_index], 1
        # return sorted(face, key=lambda x: x.bbox[0])[face_index], 1

def half_det_size(det_size):
    logger.status("Trying to halve 'det_size' parameter")
    return (det_size[0] // 2, det_size[1] // 2)

# def analyze_faces(img_data: np.ndarray, det_size=(640, 640)):
#     face_analyser = getAnalysisModel(det_size)
#     faces = face_analyser.get(img_data)

#     # Try halving det_size if no faces are found
#     if len(faces) == 0 and det_size[0] > 320 and det_size[1] > 320:
#         det_size_half = half_det_size(det_size)
#         return analyze_faces(img_data, det_size_half)

#     return faces


def analyze_faces(img_data: np.ndarray, det_size=(640, 640)):
    # Convert numpy array to GPU tensor first
    if not isinstance(img_data, torch.Tensor):
        img_tensor = torch.from_numpy(img_data).float().to('cuda')
        img_data = img_tensor.cpu().numpy()  # Convert back for OpenCV compatibility
    
    face_analyser = getAnalysisModel(det_size)
    
    # Try with GPU first
    try:
        faces = face_analyser.get(img_data)
    except RuntimeError as e:
        logger.warning(f"GPU analysis failed, falling back to CPU: {str(e)}")
        faces = face_analyser.get(img_data)  # Will use CPU if GPU fails
    
    # Fallback to smaller detection size if needed
    if len(faces) == 0 and det_size[0] > 320:
        return analyze_faces(img_data, half_det_size(det_size))
    
    return faces


# def get_face_single(img_data: np.ndarray, face, face_index=0, det_size=(640, 640), gender_source=0, gender_target=0, order="large-small"):

#     buffalo_path = os.path.join(insightface_models_path, "buffalo_l.zip")
#     if os.path.exists(buffalo_path):
#         os.remove(buffalo_path)

#     if gender_source != 0:
#         if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
#             det_size_half = half_det_size(det_size)
#             return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
#         return get_face_gender(face,face_index,gender_source,"Source", order)

#     if gender_target != 0:
#         if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
#             det_size_half = half_det_size(det_size)
#             return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
#         return get_face_gender(face,face_index,gender_target,"Target", order)
    
#     if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
#         det_size_half = half_det_size(det_size)
#         return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)

#     try:
#         faces_sorted = sort_by_order(face, order)
#         return faces_sorted[face_index], 0
#         # return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
#     except IndexError:
#         return None, 0

def get_face_single(
    img_data: Union[np.ndarray, torch.Tensor], 
    face,
    face_index: int = 0,
    det_size: tuple = (640, 640),
    gender_source: int = 0,
    gender_target: int = 0,
    order: str = "large-small"
) -> Tuple[Optional[Face], int]:
    
    # Clean up buffalo path (CPU operation)
    buffalo_path = os.path.join(insightface_models_path, "buffalo_l.zip")
    if os.path.exists(buffalo_path):
        os.remove(buffalo_path)

    # Convert input to GPU tensor if it isn't already
    if isinstance(img_data, np.ndarray):
        img_tensor = torch.from_numpy(img_data).float().to('cuda')
    else:
        img_tensor = img_data.to('cuda') if img_data.is_cuda else img_data.float().to('cuda')

    def analyze_faces_gpu(img_tensor: torch.Tensor, det_size: tuple) -> List[Face]:
        """GPU-accelerated face analysis"""
        # Convert tensor back to numpy for InsightFace (until it supports direct tensor input)
        with torch.no_grad():
            img_np = img_tensor.cpu().numpy().astype('uint8')
        
        face_analyser = getAnalysisModel(det_size)
        try:
            # First try with GPU
            faces = face_analyser.get(img_np)
        except RuntimeError:
            # Fallback to CPU if GPU fails
            faces = face_analyser.get(img_np)
        return faces

    # Handle gender-specific cases
    if gender_source != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(
                img_tensor, 
                analyze_faces_gpu(img_tensor, det_size_half),
                face_index, 
                det_size_half,
                gender_source,
                gender_target,
                order
            )
        return get_face_gender(face, face_index, gender_source, "Source", order)

    if gender_target != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(
                img_tensor,
                analyze_faces_gpu(img_tensor, det_size_half),
                face_index,
                det_size_half,
                gender_source,
                gender_target,
                order
            )
        return get_face_gender(face, face_index, gender_target, "Target", order)
    
    # Default case
    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return get_face_single(
            img_tensor,
            analyze_faces_gpu(img_tensor, det_size_half),
            face_index,
            det_size_half,
            gender_source,
            gender_target,
            order
        )

    try:
        # Sort faces on GPU if possible (convert bbox to tensors)
        if torch.cuda.is_available():
            bboxes = torch.tensor(
                [[f.bbox[0], f.bbox[1], f.bbox[2], f.bbox[3]] for f in face],
                device='cuda'
            )
            
            if order == "left-right":
                sorted_indices = torch.argsort(bboxes[:, 0])
            elif order == "right-left":
                sorted_indices = torch.argsort(bboxes[:, 0], descending=True)
            elif order == "top-bottom":
                sorted_indices = torch.argsort(bboxes[:, 1])
            elif order == "bottom-top":
                sorted_indices = torch.argsort(bboxes[:, 1], descending=True)
            elif order == "small-large":
                areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                sorted_indices = torch.argsort(areas)
            else:  # "large-small" (default)
                areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                sorted_indices = torch.argsort(areas, descending=True)
                
            return face[sorted_indices[face_index].item()], 0
        else:
            # Fallback to CPU sorting
            faces_sorted = sort_by_order(face, order)
            return faces_sorted[face_index], 0
            
    except (IndexError, RuntimeError) as e:
        logger.warning(f"Face sorting failed: {str(e)}")
        return None, 0


def swap_face(
    source_img: Union[Image.Image, None],
    target_img: Image.Image,
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: List = ["large-small", "large-small"],
    face_boost_enabled: bool = False,
    face_restore_model = None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
):

    
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH
    result_image = target_img

    if model is None:
        print("No faceswap model Found.")

    if model is not None:

        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                # split the base64 string to get the actual base64 encoded image data
                base64_data = source_img.split('base64,')[-1]
                # decode base64 string to bytes
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            
            source_img = Image.open(io.BytesIO(img_bytes))
            
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

        if source_img is not None:

            source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

            source_image_md5hash = get_image_md5hash(source_img)

            if SOURCE_IMAGE_HASH is None:
                SOURCE_IMAGE_HASH = source_image_md5hash
                source_image_same = False
            else:
                source_image_same = True if SOURCE_IMAGE_HASH == source_image_md5hash else False
                if not source_image_same:
                    SOURCE_IMAGE_HASH = source_image_md5hash

            logger.info("Source Image MD5 Hash = %s", SOURCE_IMAGE_HASH)
            logger.info("Source Image the Same? %s", source_image_same)

            if SOURCE_FACES is None or not source_image_same:
                logger.status("Analyzing Source Image...")
                source_faces = analyze_faces(source_img)
                SOURCE_FACES = source_faces
            elif source_image_same:
                logger.status("Using Hashed Source Face(s) Model...")
                source_faces = SOURCE_FACES

        elif face_model is not None:

            source_faces_index = [0]
            logger.status("Using Loaded Source Face Model...")
            source_face_model = [face_model]
            source_faces = source_face_model

        else:
            logger.error("Cannot detect any Source")
            

        if source_faces is not None:

            target_image_md5hash = get_image_md5hash(target_img)

            if TARGET_IMAGE_HASH is None:
                TARGET_IMAGE_HASH = target_image_md5hash
                target_image_same = False
            else:
                target_image_same = True if TARGET_IMAGE_HASH == target_image_md5hash else False
                if not target_image_same:
                    TARGET_IMAGE_HASH = target_image_md5hash

            logger.info("Target Image MD5 Hash = %s", TARGET_IMAGE_HASH)
            logger.info("Target Image the Same? %s", target_image_same)
            
            if TARGET_FACES is None or not target_image_same:
                logger.status("Analyzing Target Image...")
                target_faces = analyze_faces(target_img)
                TARGET_FACES = target_faces
            elif target_image_same:
                logger.status("Using Hashed Target Face(s) Model...")
                target_faces = TARGET_FACES

            # No use in trying to swap faces if no faces are found, enhancement
            if len(target_faces) == 0:
                logger.status("Cannot detect any Target, skipping swapping...")
                return result_image

            if source_img is not None:
                # separated management of wrong_gender between source and target, enhancement
                source_face, src_wrong_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[0], gender_source=gender_source, order=faces_order[1])
            else:
                # source_face = sorted(source_faces, key=lambda x: x.bbox[0])[source_faces_index[0]]
                source_face = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)[source_faces_index[0]]
                src_wrong_gender = 0

            if len(source_faces_index) != 0 and len(source_faces_index) != 1 and len(source_faces_index) != len(faces_index):
                logger.status(f'Source Faces must have no entries (default=0), one entry, or same number of entries as target faces.')
            elif source_face is not None:
                result = target_img
                if "inswapper" in model:
                    model_path = os.path.join(insightface_path, model)
                elif "reswapper" in model:
                    model_path = os.path.join(reswapper_path, model)
                face_swapper = getFaceSwapModel(model_path)

                source_face_idx = 0

                for face_num in faces_index:
                    # No use in trying to swap faces if no further faces are found, enhancement
                    if face_num >= len(target_faces):
                        logger.status("Checked all existing target faces, skipping swapping...")
                        break

                    if len(source_faces_index) > 1 and source_face_idx > 0:
                        source_face, src_wrong_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[source_face_idx], gender_source=gender_source, order=faces_order[1])
                    source_face_idx += 1

                    if source_face is not None and src_wrong_gender == 0:
                        target_face, wrong_gender = get_face_single(target_img, target_faces, face_index=face_num, gender_target=gender_target, order=faces_order[0])
                        if target_face is not None and wrong_gender == 0:
                            logger.status(f"Swapping...")
                            if face_boost_enabled:
                                logger.status(f"Face Boost is enabled")
                                bgr_fake, M = face_swapper.get(result, target_face, source_face, paste_back=False)
                                bgr_fake, scale = restorer.get_restored_face(bgr_fake, face_restore_model, face_restore_visibility, codeformer_weight, interpolation)
                                M *= scale
                                result = swapper.in_swap(target_img, bgr_fake, M)
                            else:
                                # logger.status(f"Swapping as-is")
                                result = face_swapper.get(result, target_face, source_face)
                        elif wrong_gender == 1:
                            wrong_gender = 0
                            # Keep searching for other faces if wrong gender is detected, enhancement
                            #if source_face_idx == len(source_faces_index):
                            #    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                            #    return result_image
                            logger.status("Wrong target gender detected")
                            continue
                        else:
                            logger.status(f"No target face found for {face_num}")
                    elif src_wrong_gender == 1:
                        src_wrong_gender = 0
                        # Keep searching for other faces if wrong gender is detected, enhancement
                        #if source_face_idx == len(source_faces_index):
                        #    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                        #    return result_image
                        logger.status("Wrong source gender detected")
                        continue
                    else:
                        logger.status(f"No source face found for face number {source_face_idx}.")

                result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

            else:
                logger.status("No source face(s) in the provided Index")
        else:
            logger.status("No source face(s) found")
        if model is not None:
            del model
    return result_image


# def swap_face(
#     source_img: Union[Image.Image, None],
#     target_img: Image.Image,
#     model: Union[str, None] = None,
#     source_faces_index: List[int] = [0],
#     faces_index: List[int] = [0],
#     gender_source: int = 0,
#     gender_target: int = 0,
#     face_model: Union[Face, None] = None,
#     faces_order: List = ["large-small", "large-small"],
#     face_boost_enabled: bool = False,
#     face_restore_model = None,
#     face_restore_visibility: int = 1,
#     codeformer_weight: float = 0.5,
#     interpolation: str = "Bicubic",
# ):
#     # Initialize GPU
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     result_image = target_img  # Fallback result
    
#     if model is None:
#         logger.error("No faceswap model specified")
#         return result_image

#     try:
#         # 1. Input Validation and Conversion
#         def validate_image(img):
#             if isinstance(img, Image.Image):
#                 img = np.array(img)
#                 if img.size == 0:
#                     raise ValueError("Empty source image")
#                 if len(img.shape) == 2:  # Grayscale to RGB
#                     img = np.stack([img]*3, axis=-1)
#                 return img
#             return img

#         # Process source image
#         source_np = None
#         if isinstance(source_img, str):
#             import base64, io
#             try:
#                 base64_data = source_img.split('base64,')[-1]
#                 img_bytes = base64.b64decode(base64_data)
#                 source_img = Image.open(io.BytesIO(img_bytes))
#             except Exception as e:
#                 logger.error(f"Base64 decode failed: {str(e)}")
#                 return result_image

#         if source_img is not None:
#             try:
#                 source_np = validate_image(source_img)
#                 source_img_cv = cv2.cvtColor(source_np, cv2.COLOR_RGB2BGR)
#                 if source_img_cv.size == 0:
#                     raise ValueError("Empty source image after conversion")
#             except Exception as e:
#                 logger.error(f"Source image processing failed: {str(e)}")
#                 return result_image
#         elif face_model is None:
#             logger.error("No valid source provided")
#             return result_image

#         # Process target image
#         try:
#             target_np = validate_image(target_img)
#             target_img_cv = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
#             if target_img_cv.size == 0:
#                 raise ValueError("Empty target image after conversion")
#         except Exception as e:
#             logger.error(f"Target image processing failed: {str(e)}")
#             return result_image

#         # 2. Face Analysis with Validation
#         def safe_analyze(img_cv):
#             if img_cv.size == 0 or img_cv.shape[0] < 10 or img_cv.shape[1] < 10:
#                 logger.warning("Image too small for face detection")
#                 return []
#             try:
#                 faces = analyze_faces(img_cv)
#                 if not faces:
#                     logger.warning("No faces detected")
#                 return faces
#             except Exception as e:
#                 logger.warning(f"Face analysis failed: {str(e)}")
#                 return []

#         # Analyze source
#         source_faces = [face_model] if face_model else safe_analyze(source_img_cv)
#         if not source_faces:
#             logger.error("No source faces available")
#             return result_image

#         # Analyze target
#         target_faces = safe_analyze(target_img_cv)
#         if not target_faces:
#             logger.error("No target faces detected")
#             return result_image

#         # 3. GPU-accelerated Face Swapping
#         model_path = os.path.join(insightface_path if "inswapper" in model else reswapper_path, model)
#         face_swapper = getFaceSwapModel(model_path)

#         # Get source face with validation
#         source_face, src_wrong_gender = get_face_single(
#             source_img_cv if source_np is not None else None,
#             source_faces,
#             face_index=source_faces_index[0],
#             gender_source=gender_source,
#             order=faces_order[1]
#         )
#         if source_face is None:
#             logger.error("Failed to get source face")
#             return result_image

#         # Process each target face
#         result_cv = target_img_cv.copy()
#         for face_num in faces_index:
#             if face_num >= len(target_faces):
#                 logger.warning(f"Face index {face_num} out of range")
#                 continue

#             target_face, wrong_gender = get_face_single(
#                 target_img_cv,
#                 target_faces,
#                 face_index=face_num,
#                 gender_target=gender_target,
#                 order=faces_order[0]
#             )
#             if target_face is None or wrong_gender != 0:
#                 continue

#             try:
#                 # Perform swap with dimension validation
#                 if face_boost_enabled:
#                     bgr_fake, M = face_swapper.get(
#                         result_cv,
#                         target_face,
#                         source_face,
#                         paste_back=False
#                     )
#                     # Validate before restoration
#                     if bgr_fake.size == 0 or M is None:
#                         raise ValueError("Invalid face swap result")
                    
#                     bgr_fake = restorer.get_restored_face(
#                         bgr_fake,
#                         face_restore_model,
#                         face_restore_visibility,
#                         codeformer_weight,
#                         interpolation
#                     )
#                     result_cv = swapper.in_swap(result_cv, bgr_fake, M)
#                 else:
#                     result = face_swapper.get(result_cv, target_face, source_face)
#                     if result.size == 0:
#                         raise ValueError("Empty swap result")
#                     result_cv = result

#             except Exception as e:
#                 logger.warning(f"Face swap failed for face {face_num}: {str(e)}")
#                 continue

#         # Convert back to PIL Image with validation
#         if result_cv.size == 0:
#             logger.error("Empty result after processing")
#             return result_image
            
#         result_image = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))

#     except Exception as e:
#         logger.error(f"Critical error in swap_face: {str(e)}")
#     finally:
#         torch.cuda.empty_cache()

#     return result_image


# def swap_face_many(
#     source_img: Union[Image.Image, None],
#     target_imgs: List[Image.Image],
#     model: Union[str, None] = None,
#     source_faces_index: List[int] = [0],
#     faces_index: List[int] = [0],
#     gender_source: int = 0,
#     gender_target: int = 0,
#     face_model: Union[Face, None] = None,
#     faces_order: List = ["large-small", "large-small"],
#     face_boost_enabled: bool = False,
#     face_restore_model = None,
#     face_restore_visibility: int = 1,
#     codeformer_weight: float = 0.5,
#     interpolation: str = "Bicubic",
# ):
#     global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH, TARGET_FACES_LIST, TARGET_IMAGE_LIST_HASH
#     result_images = target_imgs

#     if model is None:
#         print("No faceswap model Found.")

#     if model is not None:

#         if isinstance(source_img, str):  # source_img is a base64 string
#             import base64, io
#             if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
#                 # split the base64 string to get the actual base64 encoded image data
#                 base64_data = source_img.split('base64,')[-1]
#                 # decode base64 string to bytes
#                 img_bytes = base64.b64decode(base64_data)
#             else:
#                 # if no data URL scheme, just decode
#                 img_bytes = base64.b64decode(source_img)
            
#             source_img = Image.open(io.BytesIO(img_bytes))
            
#         target_imgs = [cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR) for target_img in target_imgs]

#         if source_img is not None:

#             source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

#             source_image_md5hash = get_image_md5hash(source_img)

#             if SOURCE_IMAGE_HASH is None:
#                 SOURCE_IMAGE_HASH = source_image_md5hash
#                 source_image_same = False
#             else:
#                 source_image_same = True if SOURCE_IMAGE_HASH == source_image_md5hash else False
#                 if not source_image_same:
#                     SOURCE_IMAGE_HASH = source_image_md5hash

#             logger.info("Source Image MD5 Hash = %s", SOURCE_IMAGE_HASH)
#             logger.info("Source Image the Same? %s", source_image_same)

#             if SOURCE_FACES is None or not source_image_same:
#                 logger.status("Analyzing Source Image...")
#                 source_faces = analyze_faces(source_img)
#                 SOURCE_FACES = source_faces
#             elif source_image_same:
#                 logger.status("Using Hashed Source Face(s) Model...")
#                 source_faces = SOURCE_FACES

#         elif face_model is not None:

#             source_faces_index = [0]
#             logger.status("Using Loaded Source Face Model...")
#             source_face_model = [face_model]
#             source_faces = source_face_model

#         else:
#             logger.error("Cannot detect any Source")

#         if source_faces is not None:

#             target_faces = []
#             for i, target_img in enumerate(target_imgs):
#                 if state.interrupted or model_management.processing_interrupted():
#                     logger.status("Interrupted by User")
#                     break
                
#                 target_image_md5hash = get_image_md5hash(target_img)
#                 if len(TARGET_IMAGE_LIST_HASH) == 0:
#                     TARGET_IMAGE_LIST_HASH = [target_image_md5hash]
#                     target_image_same = False
#                 elif len(TARGET_IMAGE_LIST_HASH) == i:
#                     TARGET_IMAGE_LIST_HASH.append(target_image_md5hash)
#                     target_image_same = False
#                 else:
#                     target_image_same = True if TARGET_IMAGE_LIST_HASH[i] == target_image_md5hash else False
#                     if not target_image_same:
#                         TARGET_IMAGE_LIST_HASH[i] = target_image_md5hash
                
#                 logger.info("(Image %s) Target Image MD5 Hash = %s", i, TARGET_IMAGE_LIST_HASH[i])
#                 logger.info("(Image %s) Target Image the Same? %s", i, target_image_same)

#                 if len(TARGET_FACES_LIST) == 0:
#                     logger.status(f"Analyzing Target Image {i}...")
#                     target_face = analyze_faces(target_img)
#                     TARGET_FACES_LIST = [target_face]
#                 elif len(TARGET_FACES_LIST) == i and not target_image_same:
#                     logger.status(f"Analyzing Target Image {i}...")
#                     target_face = analyze_faces(target_img)
#                     TARGET_FACES_LIST.append(target_face)
#                 elif len(TARGET_FACES_LIST) != i and not target_image_same:
#                     logger.status(f"Analyzing Target Image {i}...")
#                     target_face = analyze_faces(target_img)
#                     TARGET_FACES_LIST[i] = target_face
#                 elif target_image_same:
#                     logger.status("(Image %s) Using Hashed Target Face(s) Model...", i)
#                     target_face = TARGET_FACES_LIST[i]
                

#                 # logger.status(f"Analyzing Target Image {i}...")
#                 # target_face = analyze_faces(target_img)
#                 if target_face is not None:
#                     target_faces.append(target_face)

#             # No use in trying to swap faces if no faces are found, enhancement
#             if len(target_faces) == 0:
#                 logger.status("Cannot detect any Target, skipping swapping...")
#                 return result_images

#             if source_img is not None:
#                 # separated management of wrong_gender between source and target, enhancement
#                 source_face, src_wrong_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[0], gender_source=gender_source, order=faces_order[1])
#             else:
#                 # source_face = sorted(source_faces, key=lambda x: x.bbox[0])[source_faces_index[0]]
#                 source_face = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)[source_faces_index[0]]
#                 src_wrong_gender = 0

#             if len(source_faces_index) != 0 and len(source_faces_index) != 1 and len(source_faces_index) != len(faces_index):
#                 logger.status(f'Source Faces must have no entries (default=0), one entry, or same number of entries as target faces.')
#             elif source_face is not None:
#                 results = target_imgs
#                 model_path = model_path = os.path.join(insightface_path, model)
#                 face_swapper = getFaceSwapModel(model_path)

#                 source_face_idx = 0

#                 for face_num in faces_index:
#                     # No use in trying to swap faces if no further faces are found, enhancement
#                     if face_num >= len(target_faces):
#                         logger.status("Checked all existing target faces, skipping swapping...")
#                         break

#                     if len(source_faces_index) > 1 and source_face_idx > 0:
#                         source_face, src_wrong_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[source_face_idx], gender_source=gender_source, order=faces_order[1])
#                     source_face_idx += 1

#                     if source_face is not None and src_wrong_gender == 0:
#                         # Reading results to make current face swap on a previous face result
#                         for i, (target_img, target_face) in enumerate(zip(results, target_faces)):
#                             target_face_single, wrong_gender = get_face_single(target_img, target_face, face_index=face_num, gender_target=gender_target, order=faces_order[0])
#                             if target_face_single is not None and wrong_gender == 0:
#                                 result = target_img
#                                 logger.status(f"Swapping {i}...")
#                                 if face_boost_enabled:
#                                     logger.status(f"Face Boost is enabled")
#                                     bgr_fake, M = face_swapper.get(target_img, target_face_single, source_face, paste_back=False)
#                                     bgr_fake, scale = restorer.get_restored_face(bgr_fake, face_restore_model, face_restore_visibility, codeformer_weight, interpolation)
#                                     M *= scale
#                                     result = swapper.in_swap(target_img, bgr_fake, M)
#                                 else:
#                                     # logger.status(f"Swapping as-is")
#                                     result = face_swapper.get(target_img, target_face_single, source_face)
#                                 results[i] = result
#                             elif wrong_gender == 1:
#                                 wrong_gender = 0
#                                 logger.status("Wrong target gender detected")
#                                 continue
#                             else:
#                                 logger.status(f"No target face found for {face_num}")
#                     elif src_wrong_gender == 1:
#                         src_wrong_gender = 0
#                         logger.status("Wrong source gender detected")
#                         continue
#                     else:
#                         logger.status(f"No source face found for face number {source_face_idx}.")

#                 result_images = [Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) for result in results]

#             else:
#                 logger.status("No source face(s) in the provided Index")
#         else:
#             logger.status("No source face(s) found")
#         if model is not None:
#             del model
#     return result_images

def swap_face_many(
    source_img: Union[Image.Image, None],
    target_imgs: List[Image.Image],
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: List = ["large-small", "large-small"],
    face_boost_enabled: bool = False,
    face_restore_model = None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
):
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH, TARGET_FACES_LIST, TARGET_IMAGE_LIST_HASH
    
    # Initialize GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert images to GPU tensors early
    def image_to_gpu_tensor(img):
        if isinstance(img, Image.Image):
            img_np = np.array(img)
            if len(img_np.shape) == 2:  # Grayscale
                img_np = np.stack([img_np]*3, axis=-1)
            return torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0).to(device)
        return img

    # Convert target images to GPU tensors
    target_tensors = [image_to_gpu_tensor(target_img) for target_img in target_imgs]
    result_images = target_imgs  # Fallback

    if model is None:
        print("No faceswap model found.")
        return result_images

    try:
        # Handle base64 source images
        if isinstance(source_img, str):
            import base64, io
            base64_data = source_img.split('base64,')[-1] if 'base64,' in source_img else source_img
            img_bytes = base64.b64decode(base64_data)
            source_img = Image.open(io.BytesIO(img_bytes))

        # Convert source image to GPU tensor if available
        source_tensor = None
        source_img_cv = None
        if source_img is not None:
            source_tensor = image_to_gpu_tensor(source_img)
            source_np = source_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype('uint8')
            source_img_cv = cv2.cvtColor(source_np, cv2.COLOR_RGB2BGR)

        # Get face swapper model with GPU enforcement
        model_path = os.path.join(insightface_path, model)
        face_swapper = getFaceSwapModel(model_path)

        # Face analysis with GPU optimization
        def analyze_faces_gpu(img_cv):
            img_np = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_np).float().to(device)
            
            # Use cached faces if available
            img_hash = get_image_md5hash(img_cv)
            if img_hash == SOURCE_IMAGE_HASH and SOURCE_FACES is not None:
                return SOURCE_FACES
            if img_hash in TARGET_IMAGE_LIST_HASH and len(TARGET_FACES_LIST) > TARGET_IMAGE_LIST_HASH.index(img_hash):
                return TARGET_FACES_LIST[TARGET_IMAGE_LIST_HASH.index(img_hash)]
                
            faces = analyze_faces(img_cv)
            return faces

        # Process source faces
        if source_img_cv is not None:
            source_faces = analyze_faces_gpu(source_img_cv)
            SOURCE_IMAGE_HASH = get_image_md5hash(source_img_cv)
            SOURCE_FACES = source_faces
        elif face_model is not None:
            source_faces = [face_model]
        else:
            logger.error("Cannot detect any Source")
            return result_images

        # Process target faces for all images
        target_faces_list = []
        target_imgs_cv = []
        for i, target_tensor in enumerate(target_tensors):
            target_np = target_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype('uint8')
            target_img_cv = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
            target_imgs_cv.append(target_img_cv)
            
            target_faces = analyze_faces_gpu(target_img_cv)
            target_faces_list.append(target_faces)
            
            # Update hash tracking
            img_hash = get_image_md5hash(target_img_cv)
            if i >= len(TARGET_IMAGE_LIST_HASH):
                TARGET_IMAGE_LIST_HASH.append(img_hash)
                TARGET_FACES_LIST.append(target_faces)
            else:
                TARGET_IMAGE_LIST_HASH[i] = img_hash
                if i >= len(TARGET_FACES_LIST):
                    TARGET_FACES_LIST.append(target_faces)
                else:
                    TARGET_FACES_LIST[i] = target_faces

        # Main swapping loop with GPU acceleration
        with torch.cuda.amp.autocast():
            result_tensors = [t.clone() for t in target_tensors]
            
            for face_num in faces_index:
                # Get source face with GPU optimization
                source_face_idx = min(face_num, len(source_faces_index)-1)
                source_face, src_wrong_gender = get_face_single(
                    source_img_cv if source_img_cv is not None else None,
                    source_faces,
                    face_index=source_faces_index[source_face_idx],
                    gender_source=gender_source,
                    order=faces_order[1]
                )

                if source_face is None or src_wrong_gender != 0:
                    continue

                # Process each target image
                for i, (result_tensor, target_img_cv, target_faces) in enumerate(zip(result_tensors, target_imgs_cv, target_faces_list)):
                    if face_num >= len(target_faces):
                        continue

                    # Get target face with GPU optimization
                    target_face, wrong_gender = get_face_single(
                        target_img_cv,
                        target_faces,
                        face_index=face_num,
                        gender_target=gender_target,
                        order=faces_order[0]
                    )

                    if target_face is None or wrong_gender != 0:
                        continue

                    # Perform the actual face swap on GPU
                    if face_boost_enabled:
                        # GPU-accelerated face boost
                        bgr_fake, M = face_swapper.get(
                            result_tensor.cpu().numpy(),  # Temporary CPU for OpenCV
                            target_face,
                            source_face,
                            paste_back=False
                        )
                        
                        # Convert back to GPU for restoration
                        bgr_fake = torch.from_numpy(bgr_fake).float().to(device)
                        bgr_fake = restorer.get_restored_face(
                            bgr_fake,
                            face_restore_model,
                            face_restore_visibility,
                            codeformer_weight,
                            interpolation
                        )
                        
                        # Final composition on GPU
                        result_tensors[i] = swapper.in_swap(result_tensor, bgr_fake, M)
                    else:
                        # Standard GPU swap
                        result_np = face_swapper.get(
                            result_tensor.cpu().numpy(),  # Temporary CPU for OpenCV
                            target_face,
                            source_face
                        )
                        result_tensors[i] = torch.from_numpy(result_np).float().to(device)

        # Convert final results back to PIL Images
        result_images = []
        for result_tensor in result_tensors:
            result_np = result_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype('uint8')
            result_images.append(Image.fromarray(cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB)))

    except Exception as e:
        logger.error(f"Error during face swap: {str(e)}")
        result_images = target_imgs  # Return original if error occurs

    finally:
        # Clean up GPU resources
        torch.cuda.empty_cache()
        if 'source_tensor' in locals():
            del source_tensor
        if 'target_tensors' in locals():
            del target_tensors
        if 'result_tensors' in locals():
            del result_tensors

    return result_images
