# agent-get-weather-with-adk

## Setup


## Usage

DevUI(adk web)

```bash
uv run adk web
```

```txt
# prompt
What is the weather of New York?
```

Terminal(adk run)

```bash
uv run adk run src
```

API Server(adk api_server)

```bash
uv run adk api_server
```

```bash
curl -X POST http://0.0.0.0:8000/apps/weather_agent/users/u_123/sessions/s_123 \
    -H "Content-Type: application/json"
```

```bash
curl -X POST http://0.0.0.0:8000/run \
        -H "Content-Type: application/json" \
        -d '{
                "app_name": "get_weather_agent",
                "user_id": "u_123",
                "session_id": "s_123",
                "new_message": {
                    "role": "user",
                    "parts": [{
                    "text": "Hey whats the weather in new york today"
                    }]
                }
            }'
```

## FAQ

### What is the Agent2Agent Protocol (A2A)?

### What is the Agent Development Kit (ADK)?

### What is the difference between A2A and ADK?

## References
- [Agent2Agent Protocol (A2A)](https://google.github.io/A2A/)
- [Agent Development Kit](https://google.github.io/adk-docs/)
