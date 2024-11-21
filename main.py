import os
import io
import sys
import ast
import types
import base64
import traceback
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import RedirectResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO



# DEBUG = os.getenv("DEBUG", "false").lower() == "true"
DEBUG = "true"

DETECTORS_PATH = './detectors/'

app = FastAPI(
    title="FastAPI template for YOLO",
    description="",
    version="0.1",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    if DEBUG:
        # Форматируем traceback
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        
        # Возвращаем его клиенту в JSON-ответе
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal Server Error",
                "traceback": tb_str.splitlines()  # Разбиваем на строки для удобства,
            },
        )
        # return PlainTextResponse(
        #     status_code=500,
        #     content=f"Error: Internal Server Error\nTraceback:\n{tb_str}"
        # )
    
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )

# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


# Функция для динамической загрузки детектора
def load_detector(module_name, global_vars):
    # Чтение исходного кода из файла
    filepath = DETECTORS_PATH + module_name + ".py"
    with open(filepath, 'r', encoding='utf-8') as f:
        source_code = f.read()

    # Парсинг исходного кода в AST
    tree = ast.parse(source_code, filename=filepath)

    # Создание нового модуля
    module = types.ModuleType(module_name)
    module.__file__ = filepath

    # Компиляция дерева AST в исполняемый объект
    compiled_code = compile(tree, filename=filepath, mode='exec')
    
    # Обновление пространства имен модуля глобальными переменными
    module.__dict__.update(global_vars)

    # Исполнение кода в пространстве имен модуля
    exec(compiled_code, module.__dict__)

    return module


class PredictRequestModel(BaseModel):
    detector_name: str
    image: str  # base64 encoded image
    
@app.post("/run_predict")
async def run_predict(request: PredictRequestModel):
    """
    Распознать изображение с помощью выбранного детектора

    Parameters:        
        detector_name: str Имя детектора.
        image: str Изображение в формате base64.

    Returns:
        Ответ детектора в формате JSON.
    """
    function_name = "predict"
    
    detector = load_detector(request.detector_name, {"YOLO": YOLO})

    if not hasattr(detector, function_name):
        raise HTTPException(status_code=404, detail="Function not found in detector")

    function = getattr(detector, function_name)
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Убрать в клиентскую библиотеку!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    try:
        image_data = base64.b64decode(request.image)
        if len(image_data) > 5 * 1024 * 1024:  # Ограничение размера изображения до 5 МБ
            raise HTTPException(status_code=400, detail="Image size exceeds the 5MB limit")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    
    # Декодируем картинку base64 -> PIL -> np.array
    img_data = base64.b64decode(request.image)
    pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
    
    # Выполнение функции детектора с переданными параметрами
    try:
        result = function(pil_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while executing detector function: {str(e)}")

    return JSONResponse(result)


class TrainRequestModel(BaseModel):
    detector_name: str
    dataset_path: str

@app.post("/run_train")
async def run_train(request: TrainRequestModel):
    """
    Запустить функцию обучения выбранного детектора.

    Parameters:
        detector_name: str  Имя детектора.
        dataset_path: str  Пусть к датасету.

    Returns:
        Ответ детектора в формате JSON.
    """
    
    function_name = "train"
    
    detector = load_detector(request.detector_name, {"YOLO": YOLO})

    if not hasattr(detector, function_name):
        raise HTTPException(status_code=404, detail="Function not found in detector")

    function = getattr(detector, function_name)
    
    # Выполнение функции детектора с переданными параметрами
    try:
        result = function(request.detector_name, request.dataset_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while executing detector function: {str(e)}")

    return JSONResponse(result)


@app.get("/get_list")
async def get_list():
    """
    Получить список всех доступных детекторов
    
    Returns:
        Список в формате JSON.
    """
    
    # Получение списка файлов .py
    py_files = [f for f in os.listdir(detectors_path) if f.endswith('.py')]

    # Удаление расширения из имен файлов
    detectors = [os.path.splitext(f)[0] for f in py_files]

    return JSONResponse(detectors)



class MetadataRequestModel(BaseModel):
    detector_name: str
    
@app.post("/get_metadata")
async def get_metadata(request: MetadataRequestModel):
    """Получить метаданные детектора

    Args:
        request (str): Имя детектора

    Returns:
        JSON: Метаданные детектора
    """
    
    function_name = "get_metadata"
    
    detector = load_detector(request.detector_name, {"YOLO": YOLO})

    if not hasattr(detector, function_name):
        raise HTTPException(status_code=404, detail="Function not found in detector")

    function = getattr(detector, function_name)
    
    # Выполнение функции детектора с переданными параметрами
    try:
        result = function(request.detector_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while executing detector function: {str(e)}")

    return JSONResponse(result)


