import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

# 📁 Directorios
carpeta_imagenes = "dataset/images/train"
carpeta_labels = "dataset/labels/train"
os.makedirs(carpeta_labels, exist_ok=True)

# ⚙️ Parámetros
umbral_brillo = 230
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
eps = 20  # distancia máxima entre puntos del mismo grupo
min_samples = 1  # mínimo número de puntos por grupo
intensidad_teorica = 1200  # candelas teóricas para cada LED

# Parámetros de distancia (ajustado a 20 cm)
distancia_camara = 0.2  # 20 cm (0.2 metros)

# Crear una lista para almacenar los resultados
resultados = []

for archivo in os.listdir(carpeta_imagenes):
    if archivo.lower().endswith((".jpg", ".png")):
        ruta_img = os.path.join(carpeta_imagenes, archivo)
        imagen = cv2.imread(ruta_img)

        # 📐 Rotar imagen si está vertical (para estandarizar)
        if imagen.shape[0] > imagen.shape[1]:
            imagen = cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE)

        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        _, mascara = cv2.threshold(gris, umbral_brillo, 255, cv2.THRESH_BINARY)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 📍 Obtener centros de cada contorno
        centros = []
        cajas = []
        h, w = imagen.shape[:2]
        for c in contornos:
            x, y, an, al = cv2.boundingRect(c)
            if an > 3 and al > 3:
                cx = x + an / 2
                cy = y + al / 2
                centros.append([cx, cy])
                cajas.append((x, y, an, al))

        yolo_lineas = []
        led_count = 0  # Contador de LEDs detectados

        if centros:
            centros = np.array(centros)
            medidas = np.array([[an, al] for (_, _, an, al) in cajas])
            area_leds = medidas[:, 0] * medidas[:, 1]  # Área en píxeles cuadrados
            promedio = np.median(area_leds)
            tolerancia = 0.5  # 50% arriba o abajo del promedio
            idx_validos = np.where((area_leds > promedio * (1 - tolerancia)) & (area_leds < promedio * (1 + tolerancia)))[0]

            if len(idx_validos) == 0:
                print(f"{archivo} → No se detectaron LEDs tras el filtrado por dimensiones.")
                continue

            centros_filtrados = centros[idx_validos]
            cajas_filtradas = [cajas[i] for i in idx_validos]

            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centros_filtrados)

            for cluster_id in set(clustering.labels_):
                indices = np.where(clustering.labels_ == cluster_id)[0]
                if len(indices) == 0:
                    continue

                # Sumar el área de todos los LED detectados
                led_count += 1

                x_sum, y_sum, w_sum, h_sum = 0, 0, 0, 0
                for i in indices:
                    x, y, an, al = cajas_filtradas[i]
                    x_sum += x
                    y_sum += y
                    w_sum += an
                    h_sum += al
                n = len(indices)
                x_avg = x_sum / n
                y_avg = y_sum / n
                w_avg = w_sum / n
                h_avg = h_sum / n

                # Verificar si la caja no está fuera de los límites de la imagen
                if x_avg + w_avg > w or y_avg + h_avg > h or x_avg < 0 or y_avg < 0:
                    print(f"LED {cluster_id}: Caja fuera de los límites de la imagen.")
                    continue

                # Calcular el área del LED en píxeles cuadrados
                area_led = w_avg * h_avg  # en px²
                # Mostrar información sobre el área del LED en px²
                print(f"LED {cluster_id}: Área = {area_led:.2f} px²")

                # Calcular las candelas si la intensidad y el área son válidos
                try:
                    # Verifica que no haya una selección inválida de píxeles
                    intensidad_measured = np.mean(gris[int(y_avg * h):int((y_avg + h_avg) * h), int(x_avg * w):int((x_avg + w_avg) * w)])
                    if np.isnan(intensidad_measured) or intensidad_measured == 0:
                        raise ValueError("Intensidad medida inválida")
                except Exception as e:
                    print(f"Error al calcular la intensidad para el LED {cluster_id}: {e}")
                    intensidad_measured = 0

                # Validar la intensidad medida y el área
                if intensidad_measured > 0 and area_led > 0:
                    candelas = (intensidad_measured / intensidad_teorica) * (distancia_camara**2) / area_led
                    print(f"LED {cluster_id}: Candelas estimadas = {candelas:.2f} cd")
                else:
                    print(f"LED {cluster_id}: Candelas no calculadas (intensidad o área inválida).")

                # Dibujar el rectángulo que rodea el LED
                cx = (x_avg + w_avg / 2) / w
                cy = (y_avg + h_avg / 2) / h
                ancho = w_avg / w
                alto = h_avg / h
                yolo_lineas.append(f"0 {cx:.6f} {cy:.6f} {ancho:.6f} {alto:.6f}")
                cv2.rectangle(imagen, (int(x_avg), int(y_avg)), (int(x_avg + w_avg), int(y_avg + h_avg)), (0, 255, 0), 2)



        # Almacenar el resultado para la imagen actual
        resultados.append(f"{archivo} → {led_count} LED(s) detectado(s)")

        nombre_base = os.path.splitext(archivo)[0]
        ruta_label = os.path.join(carpeta_labels, f"{nombre_base}.txt")
        with open(ruta_label, "w") as f:
            f.write("\n".join(yolo_lineas))

        # 👁️ Mostrar la lista de resultados al final
        if resultados:
            print("\nResultados finales:")
            for res in resultados:
                print(res)

        # Visualización descomentada
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 8))
        plt.imshow(imagen_rgb)
        plt.title(f"{archivo} → {len(yolo_lineas)} LED(s) detectado(s)")
        plt.axis("off")
        plt.show()

        input("Presiona Enter para continuar...")

# Mostrar el resumen final de los LEDs detectados
if resultados:
    print("\nResultados finales:")
    for res in resultados:
        print(res)
