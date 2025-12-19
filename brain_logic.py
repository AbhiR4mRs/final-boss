import ollama


class FridayBrain:
    def __init__(self, owner_name="Abhiram"):
        self.owner_name = owner_name

    def generate_response(
        self,
        user_input,
        intent,
        access_mode,
        memories="",
        working_context=""
    ):
        # -------------------------------
        # SYSTEM PROMPT (RBAC + CONTEXT)
        # -------------------------------
        if access_mode == "OWNER":
            system_prompt = f"""
You are FRIDAY, a private AI assistant created by {self.owner_name}.
Status: OWNER VERIFIED.
Permissions: Full conversational access. Personal memory allowed.
Tone: Professional, loyal, intelligent (JARVIS-like).

Recent Conversation (Working Memory):
{working_context}

Long-Term Memory (Important Facts Only):
{memories}
"""
        else:
            system_prompt = f"""
You are FRIDAY operating in GUEST MODE.
Permissions: Restricted.

Rules:
- Do NOT reveal personal memories
- Do NOT confirm private details
- Politely refuse restricted actions

Recent Conversation (Working Memory):
{working_context}
"""

        # -------------------------------
        # USER PROMPT (CURRENT TURN)
        # -------------------------------
        prompt = f"""
Intent Label: {intent}
User Message: {user_input}

Respond naturally, maintaining conversational continuity.
"""

        # -------------------------------
        # STREAM RESPONSE
        # -------------------------------
        stream = ollama.generate(
            model="llama3.1",
            system=system_prompt,
            prompt=prompt,
            stream=True,
            options={
                "num_gpu": 0   # 🔥 IMPORTANT
            }
        )


        print("FRIDAY: ", end="", flush=True)
        full_response = ""

        for chunk in stream:
            token = chunk["response"]
            print(token, end="", flush=True)
            full_response += token

        print()
        return full_response
