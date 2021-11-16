from tensorflow.keras.models import load_model
model_type = 'mobilenet_v2'
model_path = f'{model_type}_saved_model'
model = load_model(model_path, compile=True)

model.summary()

