from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import TextInputSerializer
from pymongo import MongoClient  # Optional, for MongoDB connection

# MongoDB setup (optional)
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database']
collection = db['your_collection']

@api_view(['POST'])
def process_text(request):
    serializer = TextInputSerializer(data=request.data)

    if serializer.is_valid():
        user_text = serializer.validated_data['text']
        processed_text = f"Processed: {user_text}"

        # Store in MongoDB (optional)
        collection.insert_one({'original_text': user_text, 'processed_text': processed_text})

        return Response({'processed_text': processed_text})
    
    return Response(serializer.errors, status=400)