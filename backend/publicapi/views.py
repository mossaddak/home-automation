import pandas as pd

from django.shortcuts import render

from .utils import get_data


def ProcessData(request):
    fire_probability = None

    if request.method == "POST":
        MODEL, TRAIN_COLS, NUM_MEANS, CAT_MODES = get_data()

        temp = float(request.POST["temperature"])
        humidity = float(request.POST["humidity"])
        smoke = min(max(float(request.POST["smoke"]), 0), 10)
        co2 = min(max(float(request.POST["CO2"]), 0), 15)

        data_dict = {
            "Temperature_C": temp,
            "Humidity_%": humidity,
            "Smoke_Level": smoke,
            "CO2_ppm": co2,
        }

        for col in TRAIN_COLS:
            if col not in data_dict:
                if col in NUM_MEANS:
                    data_dict[col] = NUM_MEANS[col]
                elif col in CAT_MODES:
                    data_dict[col] = CAT_MODES[col]
                else:
                    data_dict[col] = 0

        input_df = pd.DataFrame([data_dict])

        prob = MODEL.predict_proba(input_df)[0][1]
        fire_probability = round(prob * 100, 2)

    return render(
        request, "fire_probability.html", {"fire_probability": fire_probability}
    )
