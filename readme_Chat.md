# Read me for chatbot db

## how it works
user sends message->fast api->pull recent chat history from db and send to gemini with new message-> gemini gives u a response->save the message and response to postgres-> return response to user

## Database Schema

Here is how the database schema is looking like:
Chat.conversations stores each chat session
chat.messages->Stores every single messages turn into conversation
chat.vlm_Assestments->stores each houses damage assestnment produced by vlm
Only the most recent chat turns are sent to Gemini so the prompt stays within quota-friendly limits.
News articles can also be imported into the database and used as cited sources in answers.
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
GEMINI_MODEL=gemini-2.5-flash-lite
### 2.Dependencies

cd backend
.venv\Scripts\activate
python -m pip install google-genai python-dotenv

### 3 Starting the Server
python -m uvicorn app.main:app 

## News article import

Use `util/import_articles.py` with a JSON file containing article objects. Each article should include:

```json
{
  "title": "Headline",
  "content": "Full article text",
  "source": "Publisher name",
  "url": "https://example.com/article",
  "published_at": "2026-04-14",
  "disaster_id": "hurricane-florence",
  "summary": "Short optional summary"
}
```

Run:

```bash
python util/import_articles.py path/to/articles.json --truncate
```

Gemini can then search those articles, cite them by title and URL, and combine them with the disaster database results.

### Import links from the chatbot

You can also paste article links into the import endpoint as raw text:

```json
{
  "text": "https://en.wikipedia.org/wiki/Hurricane_Florence https://www.weather.gov/ilm/hurricaneflorence",
  "disaster_id": "hurricane-florence"
}
```

Or send an explicit URL list:

```json
{
  "urls": [
    "https://en.wikipedia.org/wiki/Hurricane_Florence",
    "https://www.weather.gov/ilm/hurricaneflorence"
  ],
  "disaster_id": "hurricane-florence"
}
```

Then ask Gemini things like:

```text
Use the imported news articles and the disaster database to summarize Hurricane Florence, cite the article titles and URLs you used, and include the most important numbers in bullet points.
```

```text
Return JSON with keys summary, key_facts, timeline, impacts, and citations using the imported Hurricane Florence sources plus the database results.
```

You can also paste article URLs directly into a normal chat message, and the backend will import those links before Gemini answers.

### 4. Test
Go to **http://localhost:8000/docs**
test with these for given dummy data:

# conversations
docker exec -it utd-postgis psql -U utd -d utd_data -c "SELECT * FROM chat.conversations;"

# messages
docker exec -it utd-postgis psql -U utd -d utd_data -c "SELECT conversation_id, turn_index, role, LEFT(content, 60) FROM chat.messages ORDER BY conversation_id, turn_index;"

# VLM assessments
docker exec -it utd-postgis psql -U utd -d utd_data -c "SELECT id, damage_level, confidence, LEFT(description, 60) FROM chat.vlm_assessments;"
