from tensorflow.keras.models import load_model

model = load_model("model/MoBiLSTM_model.h5")

print(type(model))
print(model.input_shape)
print(model.output_shape)
