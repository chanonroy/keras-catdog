from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

IMG_SIZE = 150

model = load_model('./models/CNN.h5')

img = load_img('./test/dog.jpg', target_size=(IMG_SIZE, IMG_SIZE))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
x = x/255

value = model.predict(x)[0][0]

if value >= 0.5:
    print(value)
    print('Cat')
else:
    print(value)
    print('Dog')
