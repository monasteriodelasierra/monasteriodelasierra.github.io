import cv2
import numpy as np
import os
import json
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

def encuadra_img(dificultad='dificil'):
    """
    Encuadra imágenes en base a una imagen guía usando solo AKAZE y FLANN.
    En modo "difícil" genera combinaciones de AKAZE + FLANN con distintos parámetros.
    En modo "fácil" solo genera el primer candidato.
    
    :param dificultad: 'facil' para solo 1 candidato, 'dificil' para generar más combinaciones.
    """
    print("\nIniciando encuadre de imágenes...")
    original_dir = 'img_originales'
    processed_dir = 'img_procesadas'
    guide_dir = 'img_guia'
    candidate_dir = 'img_candidatos'
    os.makedirs(candidate_dir, exist_ok=True)

    akaze_thresholds = [0.002, 0.0015, 0.001]
    flann_params = [
        (dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), 50),
        (dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2), 75),
        (dict(algorithm=6, table_number=24, key_size=30, multi_probe_level=3), 100)
    ]

    for category in os.listdir(original_dir):
        original_folder = os.path.join(original_dir, category)
        processed_folder = os.path.join(processed_dir, category)
        guide_path = os.path.join(guide_dir, category, 'guia.jpg')
        candidate_folder = os.path.join(candidate_dir, category)
        os.makedirs(candidate_folder, exist_ok=True)
        
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)
        
        guide = cv2.imread(guide_path, cv2.IMREAD_GRAYSCALE)
        if guide is None:
            print(f"Guía no encontrada: {guide_path}, saltando...")
            continue

        processed_files = set(os.listdir(processed_folder))

        for img_file in os.listdir(original_folder):
            if img_file in processed_files:
                continue  # Saltar imágenes ya procesadas

            image_path = os.path.join(original_folder, img_file)
            print(f"Procesando encuadre de: {image_path}")
            image = cv2.imread(image_path)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            detectores = [(cv2.AKAZE_create(threshold=t), f'AKAZE({t})') for t in akaze_thresholds]
            matchers = [(cv2.FlannBasedMatcher(p, dict(checks=c)), f'FLANN({p["table_number"]}, checks={c})') for p, c in flann_params]

            candidate_idx = 0
            for detector, det_name in detectores:
                for matcher, match_name in matchers:
                    kp1, des1 = detector.detectAndCompute(guide, None)
                    kp2, des2 = detector.detectAndCompute(image_gray, None)
                    if des1 is None or des2 is None:
                        print(f"DESCARTADO {image_path}: No se encontraron descriptores con {det_name}.")
                        continue

                    matches = matcher.knnMatch(des1, des2, k=2)
                    matches = [m for m in matches if len(m) == 2]
                    good_matches = [m[0] for m in matches if m[0].distance < 0.75 * m[1].distance]
                    if len(good_matches) < 4:
                        print(f"DESCARTADO {image_path}: Pocos matches buenos con {det_name} y {match_name}.")
                        continue

                    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                    if M is None or abs(np.linalg.det(M[:2, :2])) < 0.5:
                        print(f"DESCARTADO {image_path}: Transformación inválida con {det_name} y {match_name}.")
                        continue

                    adjusted = cv2.warpAffine(image, M, (guide.shape[1], guide.shape[0]))
                    output_path = os.path.join(candidate_folder, f'{img_file[:-4]}-{candidate_idx}.jpg')
                    cv2.imwrite(output_path, adjusted)
                    print(f"GENERADO {output_path}: Usando {det_name} con {match_name}.")
                    candidate_idx += 1

                    if dificultad == 'facil':
                        break
                if dificultad == 'facil':
                    break

    print("\nProceso de encuadre completado.")


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


def generate_json_for_jpgs(directory: str):
    # Verificar si el directorio existe
    if not os.path.isdir(directory):
        print(f"Error: El directorio '{directory}' no existe.")
        return
    
    # Recorrer todas las carpetas dentro del directorio principal
    for root, dirs, files in os.walk(directory):
        # Filtrar solo archivos .jpg
        jpg_files = [f for f in files if f.lower().endswith(".jpg")]
        
        if jpg_files:
            # Crear el JSON con la lista de archivos .jpg
            json_path = os.path.join(root, "lista-imagenes.json")
            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(jpg_files, json_file, indent=2, ensure_ascii=False)
            print(f"JSON creado en: {json_path}")
        else:
            print(f"No se encontraron archivos JPG en: {root}")

# Ejemplo de uso:
# generate_json_for_jpgs("/ruta/al/directorio_X")


def main():
    print_banner("Procesamiento de imágenes de las Ventanas del Tiempo")
    #comprime_img()
    #encuadra_img()
    #selecciona_img()
    generate_json_for_jpgs("img_procesadas")
    print_banner("Proceso finalizado con éxito")

if __name__ == "__main__":
    main()
