# inferencia_pose.py (VERSIÓN FINAL, BASADA EN EL CUADERNO DE ENTRENAMIENTO)

import torch
import torch.nn as nn
import whisper
import time
import random
import math
from collections import Counter

# --- DEFINICIÓN DEL MODELO SEQ2SEQ TRANSFORMER (IDÉNTICA AL CUADERNO) ---
# Se ha copiado la arquitectura exacta de tu archivo TCC4_0 (1).ipynb

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
    def forward(self, token_embedding):
        seq_len = token_embedding.size(1)
        return self.dropout(token_embedding + self.pos_embedding[:, :seq_len, :])

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, num_heads: int, src_vocab_size: int, pose_dim: int, ffn_dim: int = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.emb_size = emb_size
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, dim_feedforward=ffn_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.tgt_pose_emb = nn.Linear(pose_dim, emb_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, nhead=num_heads, dim_feedforward=ffn_dim, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.generator = nn.Linear(emb_size, pose_dim)
    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, memory_key_padding_mask, tgt_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)
        tgt_emb = self.positional_encoding(self.tgt_pose_emb(tgt))
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return self.generator(outs)
    def encode(self, src, src_mask):
        return self.transformer_encoder(self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size)), src_mask)
    def decode(self, tgt, memory, tgt_mask):
        tgt_emb = self.positional_encoding(self.tgt_pose_emb(tgt))
        return self.transformer_decoder(tgt_emb, memory, tgt_mask)

# --- CARGA DE MODELOS Y PARÁMETROS ---

print("Cargando modelos y configuraciones...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_FILE = "best_seq2seq_pose_model_final.pth"
model = None
vocab = None
try:
    print(f"Intentando cargar el checkpoint desde '{MODEL_FILE}'...")
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    print("Checkpoint cargado.")
    
    model_args = checkpoint['model_args']
    
    model = Seq2SeqTransformer(
        num_encoder_layers=model_args['num_encoder_layers'],
        num_decoder_layers=model_args['num_decoder_layers'],
        emb_size=model_args['emb_size'],
        num_heads=model_args['num_heads'],
        src_vocab_size=model_args['src_vocab_size'],
        pose_dim=model_args['pose_dim'],
        ffn_dim=model_args['ffn_dim'],
        dropout=model_args['dropout']
    )
    
    # Cargar los pesos del modelo.
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Cargar el vocabulario guardado en el checkpoint.
    vocab = checkpoint['vocab']
    print(f"✅ ¡ÉXITO! Modelo y vocabulario desde '{MODEL_FILE}' cargados correctamente.")

except FileNotFoundError:
    print(f"❌ ERROR CRÍTICO: No se encontró el archivo del modelo en la ruta '{MODEL_FILE}'.")
except Exception as e:
    print(f"❌ ERROR CRÍTICO al cargar el modelo: {e}")

# Carga del modelo Whisper
print("Cargando modelo de Whisper...")
try:
    whisper_model = whisper.load_model("base")
    print("✅ Modelo de Whisper cargado globalmente.")
except Exception as e:
    print(f"❌ ERROR al cargar Whisper: {e}")
    whisper_model = None

# --- LÓGICA DE PROCESAMIENTO ---

PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
POSE_DIM = model_args['pose_dim'] if model else 225
MAX_POSE_SEQ_LEN = 100

def text_to_tensor(text: str):
    """Convierte un string a un tensor usando el vocabulario cargado."""
    if not vocab:
        raise ValueError("El vocabulario no está cargado.")
    return torch.tensor([vocab.get(text, UNK_IDX)], dtype=torch.long)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Lógica de traducción/inferencia adaptada del cuaderno.
def translate(model, text_tensor, max_len=MAX_POSE_SEQ_LEN):
    model.eval()
    # Añadimos los tokens de inicio y fin al texto de entrada
    src = torch.cat([torch.tensor([[BOS_IDX]]), text_tensor.view(1, -1), torch.tensor([[EOS_IDX]])], dim=1).to(device)
    with torch.no_grad():
        memory = model.encode(src, src_mask=None)
    
    # Inicializamos la secuencia de salida con el token de inicio
    ys = torch.full((1, 1, POSE_DIM), BOS_IDX, dtype=torch.float).to(device)
    
    for i in range(max_len - 1):
        with torch.no_grad():
            tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
            out = model.decode(ys, memory, tgt_mask)
            # Obtenemos la última predicción y la añadimos a la secuencia de salida
            last_pred = model.generator(out[:, -1, :]).unsqueeze(1)
            ys = torch.cat([ys, last_pred], dim=1)
            
    return ys.squeeze(0)[1:] # Quitamos el token BOS de la salida

# Nombres de los landmarks para el formato JSON.
BODY_LANDMARKS = [ "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX" ]
HAND_LANDMARKS = [ "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP" ]
TOTAL_LANDMARKS = len(BODY_LANDMARKS) + 2 * len(HAND_LANDMARKS)

def format_poses_to_json(poses_tensor, palabra_etiquetada):
    if poses_tensor is None:
        return {"error": "El tensor de poses es nulo."}

    # El tensor de salida ya es un vector de coordenadas, no de índices.
    poses_np = poses_tensor.cpu().detach().numpy()
    
    body_frames = []
    
    for frame_vector in poses_np:
        # Asegurarse de que el vector tiene la dimensión correcta
        if len(frame_vector) != POSE_DIM:
            continue
            
        frame_data = []
        # Extraer coordenadas de los landmarks del cuerpo
        for i, name in enumerate(BODY_LANDMARKS):
            idx = i * 3
            frame_data.append({"index": i, "name": name, "x": float(frame_vector[idx]), "y": float(frame_vector[idx+1]), "z": float(frame_vector[idx+2])})
        
        body_frames.append(frame_data)
        
    output_dict = {
        "fuente_video": "generado_en_directo",
        "palabra_etiquetada": palabra_etiquetada,
        "tipo_oracion": "afirmacion",
        "timestamp": time.time(),
        "pipe_data": "DATOS NO DISPONIBLES EN LA PREDICCIÓN",
        "body": body_frames,
        "body_anchored": "DATOS NO DISPONIBLES EN LA PREDICCIÓN",
        "hands": "DATOS NO DISPONIBLES EN LA PREDICCIÓN",
    }
    
    return [output_dict]