from django.shortcuts import render
from django.http import JsonResponse
from .main import *

def score_each_hour_api(request):
    lat = request.GET.get("lat")
    long = request.GET.get("long")
    date = request.GET.get("date")

    if not lat or not long:
        return JsonResponse({"error": "Missing lat or long"}, status=400)

    print(lat, long, date)

    try:
        lat = float(lat)
        long = float(long)
        date = int(date)
    except ValueError:
        return JsonResponse({"error": "Invalid lat, long or date"}, status=400)

    weather = Weather(date, lat, long)
    scores = weather.score_each_hour()
    return JsonResponse({"scores": scores})
