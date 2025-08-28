import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv

load_dotenv()

Gemini_api_key = os.getenv("GEMINI_API_KEY")

if not Gemini_api_key:
    raise ValueError("Gemini_api_key is not found!")

client = AsyncOpenAI(
    api_key=Gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True    
)

def main():
    Chatbot_agent = Agent(
        name="FAQ Agent",
        instructions="""
        You are an FAQ bot. You can only answer the following questions:
        1. What is your name?
        2. What can you do?
        3. Are you an AI?
        4. How do you work?
        5. Who made you?

        If the user asks anything else, say: "Sorry, I can only answer predefined questions."
        """,
        model=model
    )

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🤖 Welcome to the FAQ Bot!")
    print("💡 Type 'exit' anytime to quit.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" ")

    while True:
        user_question = input("🟢 Your Question: ")
        if user_question.lower() in ["exit", "quit"]:
            print("👋 Exiting... Have a great day!")
            break
        result = Runner.run_sync(Chatbot_agent, user_question, run_config=config)
        print("💬 Answer:", result.final_output)

main()