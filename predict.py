from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

IMG_SIZE = 100

model = load_model('./models/CNN-basic-100.h5')

img = load_img('./test/cat.jpg', target_size=(IMG_SIZE, IMG_SIZE))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
x = x/255

# dog = [1, 0]
# cat = [0, 1]
prediction = model.predict(x).tolist()
print(prediction[0])
