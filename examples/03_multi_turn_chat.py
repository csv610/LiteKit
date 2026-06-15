"""Example: Multi-turn conversation with LiteChat."""

from litekit import LiteChat, ModelConfig, ModelInput, ChatConfig

model_cfg = ModelConfig(model="gpt-4o-mini", temperature=0.5)
chat_cfg = ChatConfig(max_history=6)

chat = LiteChat(model_config=model_cfg, chat_config=chat_cfg)

resp1 = chat.generate_text(ModelInput(user_prompt="What is the capital of France?"))
print("Turn 1:", resp1)

resp2 = chat.generate_text(ModelInput(user_prompt="What is its most famous landmark?"))
print("Turn 2:", resp2)

print("History length:", len(chat.get_conversation_history()))

chat.reset_conversation()
print("History after reset:", len(chat.get_conversation_history()))
