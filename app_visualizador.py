# app_visualizador.py (VERSIÓN FINAL, CORREGIDA Y CON GUARDADO DE ARCHIVOS)

import gradio as gr
import os
import ffmpeg
import json
import time
from pathlib import Path

# --- Dependencias del Proyecto ---
try:
    from inferencia_pose import (
        device,
        model,
        whisper_model,
        translate,
        format_poses_to_json,
        text_to_tensor, # Se importa la función de conversión correcta
        MAX_POSE_SEQ_LEN,
        MODEL_FILE # Se importa para mensajes de error
    )
    print("✅ Dependencias de 'inferencia_pose.py' cargadas correctamente.")
except ImportError as e:
    print(f"❌ ERROR al importar desde 'inferencia_pose.py': {e}")
    exit()

# --- Creación de Carpetas ---
Path("audio_text").mkdir(exist_ok=True)
Path("movimiento predicho").mkdir(exist_ok=True)


# --- Función Principal de Procesamiento ---

def procesar_archivo_y_generar_json(ruta_archivo):
    """
    Toma un archivo, lo transcribe, genera las poses, guarda los artefactos
    y devuelve el JSON.
    """
    if ruta_archivo is None:
        raise gr.Error("Por favor, sube un archivo de audio o video primero.")

    print(f"Procesando archivo: {ruta_archivo}")

    # --- Nombres de archivo y rutas ---
    timestamp = int(time.time())
    nombre_base = Path(ruta_archivo).stem
    nombre_archivo_base = f"{nombre_base}_{timestamp}"

    ruta_audio_salida = os.path.join("audio_text", f"{nombre_archivo_base}.wav")
    ruta_texto_salida = os.path.join("audio_text", f"{nombre_archivo_base}.txt")
    ruta_json_salida = os.path.join("movimiento predicho", f"{nombre_archivo_base}.json")

    audio_path = ruta_archivo
    es_video = any(ruta_archivo.lower().endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.mkv'])

    if es_video:
        print("Archivo de video detectado. Extrayendo audio...")
        try:
            (
                ffmpeg
                .input(ruta_archivo)
                .output(ruta_audio_salida, acodec='pcm_s16le', ar='16000', ac=1)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
            audio_path = ruta_audio_salida
            print(f"Audio extraído y guardado en: '{ruta_audio_salida}'")
        except ffmpeg.Error as e:
            error_details = e.stderr.decode('utf8', errors='ignore')
            raise gr.Error(f"Error al extraer el audio: {error_details}")
    else:
        # Si es solo audio, lo guardamos en la carpeta destino
        import shutil
        shutil.copy(ruta_archivo, ruta_audio_salida)
        audio_path = ruta_audio_salida
        print(f"Archivo de audio guardado en: '{ruta_audio_salida}'")


    print("Transcribiendo audio a texto...")
    if whisper_model is None:
         raise gr.Error("El modelo Whisper no se pudo cargar.")

    try:
        result = whisper_model.transcribe(audio_path, fp16=(device.type == 'cuda'))
        texto_transcrito = result.get("text", "").strip()

        # Guardar el texto transcrito
        with open(ruta_texto_salida, "w", encoding="utf-8") as f:
            f.write(texto_transcrito)
        print(f"Texto transcrito guardado en: '{ruta_texto_salida}'")

    except Exception as e:
        raise gr.Error(f"Error durante la transcripción: {e}")

    if not texto_transcrito:
        raise gr.Error("No se pudo detectar texto en el audio.")

    print(f"Texto transcrito: '{texto_transcrito}'")
    palabras = texto_transcrito.split()
    if not palabras:
         raise gr.Error("La transcripción no produjo palabras.")

    # Se usa la primera palabra para la traducción.
    palabra_para_traducir = palabras[0].lower().strip(".,!?¡¿")
    print(f"Palabra para el modelo: '{palabra_para_traducir}'")

    if model is None:
        raise gr.Error(f"El modelo de poses no se pudo cargar. Revisa que exista el archivo '{MODEL_FILE}'.")

    print("Generando secuencia de poses...")
    try:
        # Convertir la palabra a un tensor numérico.
        palabra_tensor = text_to_tensor(palabra_para_traducir)

        # Llamar a la función `translate` corregida.
        predicted_poses_tensor = translate(model, palabra_tensor, MAX_POSE_SEQ_LEN)

    except Exception as e:
        print(f"ERROR DETALLADO al generar la pose: {e}")
        raise gr.Error(f"Error al generar la pose: {e}")

    print("Formateando salida a JSON...")
    try:
        json_output_dict = format_poses_to_json(predicted_poses_tensor, palabra_para_traducir)

        # Guardar el JSON predicho
        with open(ruta_json_salida, "w", encoding="utf-8") as f:
            json.dump(json_output_dict, f, ensure_ascii=False, indent=4)
        print(f"JSON con el movimiento predicho guardado en: '{ruta_json_salida}'")

    except Exception as e:
        print(f"ERROR DETALLADO al formatear JSON: {e}")
        raise gr.Error(f"Error al formatear a JSON: {e}")

    return f"Texto transcrito: '{texto_transcrito}'", json_output_dict


# --- Interfaz Gráfica de Usuario con Gradio ---

with gr.Blocks(theme=gr.themes.Soft(), title="Traductor de Señas a JSON") as demo:
    gr.Markdown(
        """
        # Traductor de Audio/Video a Coordenadas de Pose (JSON)
        Sube un archivo de audio o video. El sistema transcribirá la primera palabra y generará la secuencia de coordenadas de pose correspondiente.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_uploader = gr.File(label="Sube tu archivo de Audio o Video", type="filepath")
            submit_button = gr.Button("✨ Generar Coordenadas JSON", variant="primary")

            transcribed_text_output = gr.Label(label="Texto Transcrito")

        with gr.Column(scale=2):
            json_output = gr.JSON(label="Resultado en Formato JSON")

    submit_button.click(
        fn=procesar_archivo_y_generar_json,
        inputs=file_uploader,
        outputs=[transcribed_text_output, json_output]
    )

    gr.Markdown("---")
    gr.Markdown("### ¿Cómo funciona?")
    gr.Markdown("1.  **Carga:** Sube un archivo de video o audio.")
    gr.Markdown("2.  **Procesa:** El sistema extrae el audio y usa **Whisper** para transcribirlo.")
    gr.Markdown("3.  **Traduce:** La primera palabra se convierte en un tensor y se envía al modelo **Seq2Seq Transformer**.")
    gr.Markdown("4.  **Muestra:** Las coordenadas 3D de la pose se muestran en formato JSON.")

# Lanzar la aplicación
if __name__ == "__main__":
    demo.launch(share=True)