import base64
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse

import numpy as np
from keras import models 
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
#from .prediction import lung_predict


def lung_predict(model):
  

    img=load_img('C:\\Users\\navee\\Desktop\\lung-disease\\backend\\image.jpg',target_size=(224,224))

    x=img_to_array(img)

    x=np.expand_dims(x, axis=0)

    img_data=preprocess_input(x)

    classes=model.predict(img_data)

    result=int(classes[0][0])

    if result==0:

        print("Person is Affected By PNEUMONIA")

        return "Person is Affected By PNEUMONIA"
    else:
        print("Result is Normal")

        return "Result is Normal"

   


class UploadImageView(APIView):
   
    def post(self, request, format=None):
        # decode image data from request body
        
        model = models.load_model('C:\\Users\\navee\\Desktop\\lung-disease\\backend\\prediction_app\\chest_xray.h5')
        imageDataUrl = request.data.get('imageDataUrl')
        print(imageDataUrl)

        try:
            imageData = base64.b64decode(imageDataUrl.split(',')[1])
        except IndexError:
            return JsonResponse({'error': 'Invalid image data'})

        # save image to file
        with open('image.jpg', 'wb') as f:
            f.write(imageData)

        result = lung_predict(model)   

        re_result = {'success': result}

        # do something with the image (e.g., process it using your AI model)

        # return success response
        return JsonResponse(re_result)




