import cv2
import numpy as np
import os
import time
import threading
from PIL import Image

def print_banner(text):
    """Imprime un texto enmarcado con colores."""
    border = "=" * (len(text) + 4)
    print(f"\033[94m{border}\033[0m")
    print(f"\033[94m| {text} |\033[0m")
    print(f"\033[94m{border}\033[0m")

def comprime_img():
    """Convierte imágenes a formato JPG, comprime su tamaño y las mueve a img_originales."""
    print("\n")
    print("Iniciando compresión de imágenes...")
    input_dir = 'img_nuevas'
    output_dir = 'img_originales'
    quality = 85
    count = 0
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, input_dir)
            output_folder = os.path.join(output_dir, rel_path)
            os.makedirs(output_folder, exist_ok=True)
            
            output_path = os.path.join(output_folder, os.path.splitext(file)[0] + '.jpg')
            print(f"Procesando: {input_path} -> {output_path}")
            
            with Image.open(input_path) as img:
                img = img.convert("RGB")
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
            os.remove(input_path)
            count += 1
            print(f"Imagen comprimida y guardada: {output_path}")
    
    print(f"Total de imágenes comprimidas: {count}")

def encuadra_img():
    """Encuadra imágenes de img_originales que no están en img_procesadas usando AKAZE con diferentes parámetros avanzados."""
    print("\n")
    print("Iniciando encuadre de imágenes...")
    original_dir = 'img_originales'
    processed_dir = 'img_procesadas'
    guide_dir = 'img_guia'
    candidate_dir = 'img_candidatos'
    os.makedirs(candidate_dir, exist_ok=True)
    
    for category in os.listdir(original_dir):
        original_folder = os.path.join(original_dir, category)
        processed_folder = os.path.join(processed_dir, category)
        guide_path = os.path.join(guide_dir, category, 'guia.jpg')
        candidate_folder = os.path.join(candidate_dir, category)
        os.makedirs(candidate_folder, exist_ok=True)
        
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)
        
        for img_file in os.listdir(original_folder):
            if img_file.lower() == 'guia.jpg' or img_file in os.listdir(processed_folder):
                continue  # Evitar procesar la imagen guía y las ya procesadas
            
            image_path = os.path.join(original_folder, img_file)
            print(f"Procesando encuadre de: {image_path}")
            guide = cv2.imread(guide_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(image_path)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            for i in range(10):
                start_time = time.time()
                try:
                    detector = cv2.AKAZE_create(
                        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                        descriptor_size=0,
                        descriptor_channels=3,
                        threshold=0.0020 - i * 0.0002,
                        nOctaves=4,
                        nOctaveLayers=4,
                        diffusivity=cv2.KAZE_DIFF_PM_G2
                    )
                    kp1, des1 = detector.detectAndCompute(guide, None)
                    kp2, des2 = detector.detectAndCompute(image_gray, None)
                    
                    if des1 is None or des2 is None:
                        print(f"No se encontraron descriptores en {image_path} con threshold {0.0020 - i * 0.0002}. Saltando candidato...")
                        continue
                    
                    matcher = cv2.BFMatcher()
                    matches = matcher.knnMatch(des1, des2, k=2)
                    
                    matches = [m for m in matches if len(m) == 2]
                    matches = sorted(matches, key=lambda x: x[0].distance)[:min(50, len(matches))]
                    good_matches = [m[0] for m in matches if m[0].distance < (0.77 - i * 0.01) * m[1].distance]
                    
                    if len(good_matches) < 4:
                        print(f"No suficientes good_matches en {image_path} con threshold {0.0020 - i * 0.0002}. Saltando candidato...")
                        continue
                    
                    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
                    adjusted = cv2.warpAffine(image, M, (guide.shape[1], guide.shape[0]))
                    output_path = os.path.join(candidate_folder, f'{img_file[:-4]}-{i}.jpg')
                    cv2.imwrite(output_path, adjusted)
                    elapsed_time = time.time() - start_time
                    print(f"Imagen candidata generada: {output_path} en {elapsed_time:.2f} segundos con threshold {0.0020 - i * 0.0002} y distancia {0.77 - i * 0.01}")
                except Exception as e:
                    print(f"Error al procesar candidato {i} para {image_path}: {e}. Continuando con el siguiente.")


def selecciona_img():
    """Selecciona una imagen de img_candidatos y la mueve a img_procesadas o elimina todas si no se elige ninguna."""
    print("\n")
    print("Iniciando selección de imágenes...")
    candidate_dir = 'img_candidatos'
    processed_dir = 'img_procesadas'
    os.makedirs(processed_dir, exist_ok=True)
    
    for category in os.listdir(candidate_dir):
        candidate_folder = os.path.join(candidate_dir, category)
        processed_folder = os.path.join(processed_dir, category)
        os.makedirs(processed_folder, exist_ok=True)
        
        img_prefixes = set("-".join(f.split('-')[:-1]) for f in os.listdir(candidate_folder) if f.endswith('.jpg'))
        img_prefixes = [prefix for prefix in img_prefixes if prefix]  # Filtra prefijos vacíos
        
        for img_prefix in img_prefixes:
            if not any(img_prefix in f for f in os.listdir(processed_folder)):
                print(f"Selecciona una imagen para {img_prefix} en {category}:")
                candidates = sorted([f for f in os.listdir(candidate_folder) if f.startswith(img_prefix)])
                for idx, img_file in enumerate(candidates):
                    print(f"{idx}: {img_file}")
                print("n: No seleccionar ninguna y eliminar todas")
                
                choice = input("Número de la imagen seleccionada: ")
                
                if choice.lower() == "n":
                    print(f"Eliminando todos los candidatos para {img_prefix}...")
                    for img in candidates:
                        os.remove(os.path.join(candidate_folder, img))
                        print(f"Imagen eliminada: {img}")
                else:
                    choice = int(choice)
                    selected_img = candidates[choice]
                    
                    src_path = os.path.join(candidate_folder, selected_img)
                    dst_path = os.path.join(processed_folder, img_prefix + '.jpg')
                    os.rename(src_path, dst_path)
                    print(f"Imagen seleccionada y movida: {dst_path}")
                    
                    for img in candidates:
                        if img != selected_img:  # Evitar eliminar la imagen ya movida
                            os.remove(os.path.join(candidate_folder, img))
                            print(f"Imagen eliminada: {img}")

def main():
    print_banner("Procesamiento de imágenes de las Ventanas del Tiempo")
    comprime_img()
    encuadra_img()
    selecciona_img()
    print_banner("Proceso finalizado con éxito")

if __name__ == "__main__":
    main()
