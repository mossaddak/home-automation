import pandas as pd

from django.shortcuts import render

from .utils import get_data

GLOBAL_MODEL = get_data()


def ProcessData(request):
    fire_probability = None

    if request.method == "POST":
        temp = float(request.POST.get("temperature", 0))
        humidity = float(request.POST.get("humidity", 0))
        smoke = float(request.POST.get("smoke", 0))
        gas = float(request.POST.get("gas", 0))

        data_dict = {
            "Temperature_C": temp,
            "Humidity_%": humidity,
            "Smoke_Level": smoke,
            "Gas_Level": gas,
            "Room_Type": "Bedroom",
            "Sound_Level_dB": 0,
            "CO2_ppm": 0,
            "Smart_Door_Status": 0,
            "Window_Status": 0,
            "Light_Intensity_lux": 0,
            "Motion_Detected": 0,
            "Fan_Status": 0,
            "Day_Night": "Day",
            "Occupancy": 0,
            "Smart_LED_On": 0,
            "Power_Consumption_W": 0,
            "AC_Status": 0,
        }

        input_data = pd.DataFrame([data_dict])

        try:
            prob = GLOBAL_MODEL.predict_proba(input_data)[0][1]
            fire_probability = round(prob * 100, 2)
        except Exception as e:
            print(f"Prediction error: {e}")
            fire_probability = 0

    return render(request, "fire_probability.html", {"fire_probability": fire_probability})
