import requests
import json
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

SYSTEM_CONTEXT = os.getenv("SYSTEM_PROMPT")

def ask_concierge():
    print("--- NCCS Data Concierge (4-Message Memory) ---")
    print("Type 'exit' or 'quit' to stop.\n")

    # Initialize a list to hold the chat history
    # This stays OUTSIDE the while loop so it doesn't reset
    message_history = []

    while True:
        user_prompt = input("Your Prompt: ")

        if user_prompt.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        if not user_prompt.strip():
            continue

        # 1. Add User message to the full history
        message_history.append({"role": "user", "content": user_prompt})

        # 2. Slice the history to get only the last 4 messages
        # We then add the SYSTEM_CONTEXT at the very front
        recent_history = message_history[-4:]
        payload_messages = [{"role": "system", "content": SYSTEM_CONTEXT}] + recent_history

        try:
            print("Thinking...")
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": "deepseek/deepseek-r1-0528:free",
                    "messages": payload_messages # Send the windowed history
                }),
                timeout=60 
            )

            data = response.json()
            
            if "choices" in data:
                message = data["choices"][0]["message"]
                content = message.get("content", "")

                if content:
                    print(f"\nAI Answer:\n{content}\n")
                    # 3. Add AI response to history so it can "remember" its own answers
                    message_history.append({"role": "assistant", "content": content})
                else:
                    print("AI returned an empty response.")
            else:
                print("API Error:", data.get("error", "Unknown Error"))

        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

if __name__ == "__main__":
    if not API_KEY or not SYSTEM_CONTEXT:
        print("Error: Check your .env for API_KEY and SYSTEM_PROMPT.")
    else:
        ask_concierge()