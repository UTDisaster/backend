# Read me for chatbot db

## how it works
user sends message->fast api->pull chat history from db and send to gemini with new message-> gemini gives u a response->save the message and response to postgres-> return response to user

## Database Schema

Here is how the database schema is looking like:
Chat.conversations stores each chat session
chat.messages->Stores every single messages turn into conversation
chat.vlm_Assestments->stores each houses damage assestnment produced by vlm
Every message to gemini includes full chat history
API endpoints:
POST /chat/message->create/load conversations
GET /chat/conversations->


## API Endpoints

### `POST /chat/message`
create/load conversations

### `GET /chat/conversations`
all conversations with their title and last reply.

### `GET /chat/conversations/{conversation_id}`
give the conversation id and returns a full conversation 
### `DELETE /chat/conversations/{conversation_id}`
Deletes a conversation and all its messages

## Setup

### backend/.env:

DATABASE_URL=postgresql+psycopg2://utd:utdpass@127.0.0.1:5432/utd_data
GEMINI_API_KEY=your_gemini_api_key_here
### 2.Dependencies

cd backend
.venv\Scripts\activate
python -m pip install google-genai python-dotenv

### 3 Starting the Server
python -m uvicorn app.main:app 

### 4. Test
Go to **http://localhost:8000/docs**
test with these for given dummy data:

# conversations
docker exec -it utd-postgis psql -U utd -d utd_data -c "SELECT * FROM chat.conversations;"

# messages
docker exec -it utd-postgis psql -U utd -d utd_data -c "SELECT conversation_id, turn_index, role, LEFT(content, 60) FROM chat.messages ORDER BY conversation_id, turn_index;"

# VLM assessments
docker exec -it utd-postgis psql -U utd -d utd_data -c "SELECT id, damage_level, confidence, LEFT(description, 60) FROM chat.vlm_assessments;"
