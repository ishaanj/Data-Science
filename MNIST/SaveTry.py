from keras.models import Sequential

model = Sequential()

json_string = model.to_json()
open('Models/arch1_test.json', 'w').write(json_string)
model.save_weights('Models/arch1_weight_test.h5')
