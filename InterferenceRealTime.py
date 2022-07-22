import cv2
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# tflite_model_file = 'model.tflite'
# with open(tflite_model_file, 'rb') as fid:
#     tflite_model = fid.read()
    
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
  
# define a video capture object
vid = cv2.VideoCapture(1)
class_names = ['Dua Ribu', 'Dua Puluh Ribu', 'Lima Ribu']

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    
    frame = cv2.resize(frame, (150, 150))
    x=image.img_to_array(frame)
    x /= 255
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])

    interpreter.set_tensor(input_index, images)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)
    predicted_label = np.argmax(prediction)
    if 100*np.max(prediction) > 80:
        print(f'{class_names[predicted_label]} {100*np.max(prediction)}')
    
    cv2.imshow('frame', frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
print('akhir')