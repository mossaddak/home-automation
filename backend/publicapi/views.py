from rest_framework.views import APIView
from rest_framework.response import Response


class ProcessData(APIView):

    def post(self, request):
        return Response({"result": 100}, status=201)