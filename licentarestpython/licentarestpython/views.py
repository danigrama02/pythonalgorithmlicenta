from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import weatherPredictionModel

model = weatherPredictionModel.load_model("model_v13.pkl")

@api_view(['POST'])
def get_weather(request):
    data = request.data
    locations = data
    print("Weather request recieved for locations " + str(locations))
    print("Computeing..")
    predictions = []
    for location in locations:
        predictions.append(weatherPredictionModel.predict(model,location))
    return Response( {"predictions" : predictions})