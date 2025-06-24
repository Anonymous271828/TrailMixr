from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
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

    weather = Weather(date, lat, long, 0)
    scores = weather.score_each_hour()
    return JsonResponse({"scores": scores})


def test_get_all_trails(request):
    c = Calculate()
    return JsonResponse({"trails": c.get_trails()})


@csrf_exempt
def upload_file(request):
    print("Upload file called")
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_contents = uploaded_file.read().decode('utf-8')  # adjust encoding if needed

        result = main(file_contents)
        print(result)
        return JsonResponse(result)

    return JsonResponse({'error': 'Invalid request'}, status=400)
