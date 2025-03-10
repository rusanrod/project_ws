#!/usr/bin/env python3

import torch
from ultralytics import YOLO
from torchvision import models, transforms
import cv2
import numpy as np
import os
from scipy.spatial.distance import cosine
import rospkg

rp = rospkg.RosPack()
pack_path = rp.get_path("obj_matching")
path = os.path.join(pack_path, "ref_imgs")

# Configuracion de CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Carga el modelo YOLOv8 preentrenado
# model = YOLO('yolov8n.pt').to(device)

model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device)
model.eval()

'''def detect_objects(image):
    results = model(image)
    detections = []
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            detections.append((x1, y1, x2, y2))
    return detections

# Cargar extractor de embeddings (MobileNetV3 Small)
feature_extractor = models.mobilenet_v3_small(pretrained =True).to(device)
# Quitar capa de clasificación
feature_extractor = torch.nn.Sequential(*list(list(feature_extractor.children())[:-1]))
feature_extractor.eval()

# Transformaciones para entrada al modelo
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Obtener embedding de una imagen
def get_embedding(image):
    with torch.no_grad():
        return feature_extractor(preprocess_image(image)).squeeze().cpu().numpy()

# Comparar embeddings con similitud coseno
def compare_embeddings(emb1, emb2):
    return 1 - cosine(emb1, emb2)

def load_reference_images(folder):
    ref_embeddings = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        image = cv2.imread(filepath)
        if image is not None:
            ref_embeddings.append(get_embedding(image))
    return ref_embeddings


# Main function

# def match_object(ref_img, target_img):
#     ref_embedding = get_embedding(ref_img)
#     detections = detect_objects(target_img)
#     matches = []

#     for (x1, y1, x2, y2) in detections:
#         obj_crop = target_img[y1:y2, x1:x2]
#         obj_embedding = get_embedding(obj_crop)
#         similarity = compare_embeddings(ref_embedding, obj_embedding)
#         if similarity > 0.7:
#             matches.append((x1, y1, x2, y2), similarity)

#     return matches

# Función principal con webcam
def match_object(reference_folder):
    cap = cv2.VideoCapture(0)
    reference_embeddings = load_reference_images(reference_folder)
    print(reference_embeddings)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detect_objects(frame)
        for (x1, y1, x2, y2) in detections:
            obj_crop = frame[y1:y2, x1:x2]
            obj_embedding = get_embedding(obj_crop)
            
            for ref_embedding in reference_embeddings:
                similarity = compare_embeddings(ref_embedding, obj_embedding)
                if similarity > 0.6:  # Umbral de similitud
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Match {similarity:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Object Matching", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Ejecutar con carpeta de imágenes de referencia
match_object("./ref_imgs")
'''


# Transformaciones para entrada al modelo
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Obtener segmentos de una imagen
def segment_objects(image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    return output_predictions

# Obtener bounding boxes a partir de los segmentos
def get_bounding_boxes(segmentation):
    contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(contour) for contour in contours]
    return [(x, y, x + w, y + h) for (x, y, w, h) in boxes]

# Cargar extractor de embeddings (MobileNetV3 Small)
feature_extractor = models.mobilenet_v3_small(pretrained=True).to(device)
# Quitar capa de clasificación
feature_extractor = torch.nn.Sequential(*list(list(feature_extractor.children())[:-1]))
feature_extractor.eval()

# Obtener embedding de una imagen
def get_embedding(image):
    with torch.no_grad():
        return feature_extractor(preprocess_image(image)).squeeze().cpu().numpy()

# Comparar embeddings con similitud coseno
def compare_embeddings(emb1, emb2):
    return 1 - cosine(emb1, emb2)

def load_reference_images(folder):
    ref_embeddings = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        image = cv2.imread(filepath)
        if image is not None:
            ref_embeddings.append(get_embedding(image))
    return ref_embeddings

# Función principal con webcam
def match_object(reference_folder):
    cap = cv2.VideoCapture(0)
    reference_embeddings = load_reference_images(reference_folder)
    # print(reference_embeddings)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        segmentation = segment_objects(frame)
        detections = get_bounding_boxes(segmentation)
        for (x1, y1, x2, y2) in detections:
            obj_crop = frame[y1:y2, x1:x2]
            obj_embedding = get_embedding(obj_crop)
            
            for ref_embedding in reference_embeddings:
                similarity = compare_embeddings(ref_embedding, obj_embedding)
                if similarity > 0.6:  # Umbral de similitud
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Match {similarity:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Object Matching", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Ejecutar con carpeta de imágenes de referencia
match_object(path)