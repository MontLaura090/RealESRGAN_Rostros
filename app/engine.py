import sys
import os
import threading
import cv2
import numpy as np
import torch
from pathlib import Path
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Añadir la ruta de CodeFormer al sistema para futuras referencias
CODEFORMER_PATH = Path(__file__).resolve().parent.parent / "CodeFormer"
if str(CODEFORMER_PATH) not in sys.path:
    sys.path.append(str(CODEFORMER_PATH))

try:
    CODEFORMER_AVAILABLE = True
except ImportError:
    CODEFORMER_AVAILABLE = False

_lock = threading.Lock()
_engine = None
_device = None

def get_engine():
    global _engine, _device
    with _lock:
        if _engine is None:
            path = os.getenv('MODEL_PATH', 'weights/RealESRGAN_x4plus.pth')
            if not Path(path).is_file():
                raise RuntimeError(
                    f'Pesos no encontrados: {path}. Descargue RealESRGAN_x4plus.pth desde los releases oficiales.'
                )
            _device = 'cuda' if torch.cuda.is_available() else 'cpu'
            net = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )
            _engine = RealESRGANer(
                scale=4,
                model_path=path,
                model=net,
                tile=256,
                tile_pad=15,
                pre_pad=0,
                half=(_device == 'cuda'),
                gpu_id=0 if _device == 'cuda' else None
            )
        return _engine, _device

def calculate_sharpness(img_bgr):
    """Calcula la Varianza del Laplaciano para medir la nitidez."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

def calculate_psnr(img1, img2):
    """Calcula el PSNR (Peak Signal-to-Noise Ratio) reescalando img1 al tamaño de img2."""
    h, w = img2.shape[:2]
    img1_resized = cv2.resize(img1, (w, h), interpolation=cv2.INTER_CUBIC)
    mse = np.mean((img1_resized.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return round(float(20 * np.log10(255.0 / np.sqrt(mse))), 2)

def restore(raw: bytes, scale: float = 4.0):
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('La imagen subida no es válida o está corrupta.')
    if img.shape[0] * img.shape[1] > 12_000_000:
        raise ValueError('La dimensión de la imagen excede el límite permitido (12 Megapíxeles).')

    engine, device = get_engine()
    with _lock:
        try:
            out, _ = engine.enhance(img, outscale=scale)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and device == 'cuda':
                torch.cuda.empty_cache()
                raise RuntimeError('Memoria VRAM insuficiente en GPU. Intenta usar una escala menor.')
            raise e

    # Cálculo de Métricas Objetivas
    laplacian_before = calculate_sharpness(img)
    laplacian_after = calculate_sharpness(out)
    psnr_score = calculate_psnr(img, out)

    ok, enc_out = cv2.imencode('.png', out)
    ok_in, enc_in = cv2.imencode('.png', img)
    if not ok or not ok_in:
        raise RuntimeError('Error al codificar las imágenes.')

    metrics = {
        'laplacian_before': laplacian_before,
        'laplacian_after': laplacian_after,
        'psnr': psnr_score
    }
    
    return enc_out.tobytes(), enc_in.tobytes(), device, (img.shape[1], img.shape[0]), (out.shape[1], out.shape[0]), metrics

def edit(raw: bytes, brightness: int = 0, contrast: float = 1.0, filter_name: str = 'none'):
    arr = np.frombuffer(raw, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError('La imagen subida no es válida o está corrupta.')

    # Ajuste de Brillo y Contraste
    im = cv2.convertScaleAbs(im, alpha=float(contrast), beta=int(brightness))

    # Filtros avanzados
    if filter_name == 'denoise':
        im = cv2.fastNlMeansDenoisingColored(im, None, 5, 5, 7, 21)
    elif filter_name == 'sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        im = cv2.filter2D(im, -1, kernel)
    elif filter_name == 'blur':
        im = cv2.GaussianBlur(im, (5, 5), 0)
    elif filter_name == 'grayscale':
        im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    ok, enc = cv2.imencode('.png', im)
    if not ok:
        raise RuntimeError('Error al procesar los cambios de edición.')
    return enc.tobytes()