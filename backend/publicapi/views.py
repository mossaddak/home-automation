import pandas as pd

from django.shortcuts import render

from .utils import get_data


def ProcessData(request):
    fire_probability = None

    if request.method == "POST":
        MODEL, TRAIN_COLS, NUM_MEANS, CAT_MODES = get_data()

        payload = {
            "Temperature_C": float(request.POST["temperature"]),
            "Humidity_%": float(request.POST["humidity"]),
            "Smoke_Level": min(max(float(request.POST["smoke"]), 0), 10),
            "CO2_ppm": min(max(float(request.POST["CO2"]), 0), 15),
        }

        for col in TRAIN_COLS:
            if col not in payload:
                if col in NUM_MEANS:
                    payload[col] = NUM_MEANS[col]
                elif col in CAT_MODES:
                    payload[col] = CAT_MODES[col]
                else:
                    payload[col] = 0

        input_df = pd.DataFrame([payload])
        prob = MODEL.predict_proba(input_df)[0][1]
        fire_probability = round(prob * 100, 2)

    return render(
        request, "fire_probability.html", {"fire_probability": fire_probability}
    )
