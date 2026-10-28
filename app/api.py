import os
import io
import base64
import uuid
import datetime
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
from dotenv import load_dotenv

try:
    from app.engine import restore, edit
except ModuleNotFoundError:
    from engine import restore, edit

load_dotenv()

app = FastAPI(title='Real-ESRGAN Facial Image Restoration', version='1.2')

client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017'), serverSelectionTimeoutMS=4000)
db = client[os.getenv('MONGO_DB', 'face_restoration')]
db.jobs.create_index([('created_at', DESCENDING)])

MAX = int(os.getenv('MAX_UPLOAD_MB', '10')) * 1024 * 1024

def save(raw_out: bytes, raw_in: bytes, original: str, device: str, old: tuple, new: tuple, scale: float, elapsed: float, metrics: dict):
    from gridfs import GridFS
    fs = GridFS(db)
    fid_out = fs.put(raw_out, filename='restored.png', content_type='image/png')
    fid_in = fs.put(raw_in, filename='original.png', content_type='image/png')
    try:
        doc = {
            'created_at': datetime.datetime.now(datetime.timezone.utc),
            'filename': original,
            'gridfs_id': fid_out,
            'gridfs_original_id': fid_in,
            'device': device,
            'input_size': old,
            'output_size': new,
            'scale': scale,
            'seconds': elapsed,
            'metrics': metrics,
            'status': 'completed'
        }
        return str(db.jobs.insert_one(doc).inserted_id)
    except Exception:
        fs.delete(fid_out)
        fs.delete(fid_in)
        raise

@app.get('/health')
def health():
    try:
        client.admin.command('ping')
        mongo = 'ok'
    except Exception as e:
        mongo = str(e)
    return {'api': 'ok', 'mongodb': mongo}

@app.post('/restore')
async def reconstruct(file: UploadFile = File(...), scale: float = Form(4.0)):
    if not (1.0 <= scale <= 4.0):
        raise HTTPException(400, 'Escala permitida únicamente entre 1 y 4')
    if file.content_type not in ('image/jpeg', 'image/png', 'image/webp'):
        raise HTTPException(415, 'Formato no soportado. Debe ser JPEG, PNG o WebP')
    
    raw = await file.read(MAX + 1)
    if len(raw) > MAX:
        raise HTTPException(413, 'El archivo supera el tamaño máximo permitido (10MB)')
    
    try:
        t = time.perf_counter()
        result_out, result_in, device, old, new, metrics = restore(raw, scale)
        job_id = save(result_out, result_in, file.filename, device, old, new, scale, round(time.perf_counter() - t, 3), metrics)
        return {
            'id': job_id,
            'device': device,
            'input_size': old,
            'output_size': new,
            'metrics': metrics,
            'image_base64': base64.b64encode(result_out).decode('ascii')
        }
    except ValueError as ve:
        raise HTTPException(400, str(ve))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post('/edit')
async def editing(
    file: UploadFile = File(...),
    brightness: int = Form(0),
    contrast: float = Form(1.0),
    filter_name: str = Form('none'),
    save_result: bool = Form(True)
):
    if not (-100 <= brightness <= 100 and 0.2 <= contrast <= 3.0):
        raise HTTPException(400, 'Parámetros fuera de rango')
    if filter_name not in ('none', 'denoise', 'sharpen', 'blur', 'grayscale'):
        raise HTTPException(400, 'Filtro no válido')
    
    raw = await file.read(MAX + 1)
    if len(raw) > MAX:
        raise HTTPException(413, 'El archivo supera el tamaño máximo')
    
    try:
        result = edit(raw, brightness, contrast, filter_name)
        fid = None
        if save_result:
            from gridfs import GridFS
            fs = GridFS(db)
            fid = fs.put(result, filename='edited.png', content_type='image/png')
            db.edits.insert_one({
                'created_at': datetime.datetime.now(datetime.timezone.utc),
                'gridfs_id': fid,
                'brightness': brightness,
                'contrast': contrast,
                'filter': filter_name
            })
        return {
            'saved': bool(fid),
            'image_base64': base64.b64encode(result).decode('ascii')
        }
    except ValueError as ve:
        raise HTTPException(400, str(ve))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get('/history')
def history(limit: int = 50):
    res = []
    for d in db.jobs.find().sort('created_at', -1).limit(min(max(limit, 1), 200)):
        m = d.get('metrics', {})
        res.append({
            'id': str(d['_id']),
            'created_at': d['created_at'].isoformat(),
            'filename': d['filename'],
            'device': d['device'],
            'scale': d['scale'],
            'seconds': d['seconds'],
            'input_size': d['input_size'],
            'output_size': d['output_size'],
            'laplacian_before': m.get('laplacian_before', 0),
            'laplacian_after': m.get('laplacian_after', 0),
            'psnr': m.get('psnr', 0)
        })
    return res

@app.get('/image/{id}')
def image(id: str, type: str = 'restored'):
    try:
        from gridfs import GridFS
        doc = db.jobs.find_one({'_id': ObjectId(id)})
        if not doc:
            raise HTTPException(404, 'Imagen no encontrada')
        
        target_id = doc.get('gridfs_original_id') if type == 'original' else doc['gridfs_id']
        if not target_id:
            target_id = doc['gridfs_id']
            
        return Response(GridFS(db).get(target_id).read(), media_type='image/png')
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, 'ID no válido')