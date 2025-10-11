from ollama import chat, ChatResponse
#https://ollama.readthedocs.io/en/api/#generate-a-chat-completion

#https://labs.jstor.org/developers/ look at later

class Ollama:
    def __init__(self, model="llama3.2"):
        self.model = model
        self.conversation_history = []
    
    def add_msg(self, role, content):
        self.conversation_history.append({
            "role": role, #the role of the message, either *system*, *user*, *assistant*, or *tool*
            "content": content
        })

    def stream_response(self):
        response: ChatResponse = chat(
            model='llama3.2',
            messages=self.conversation_history,
            stream=False #Tools requires stream to be set to false
            #tool_calls=""
            #format=""
        )

        reply = response.message.content
        print("Here:" + reply)
        print()

        self.add_msg("assistant", reply)

        return reply
    
    def reset(self):
        self.conversation_history = []

def main():
    inputs = ["Hello", "Why is the sky blue?", "What was my previous question?"]
    ollama = Ollama()
    for i in inputs:
        print("You: " + i)
        print()
        ollama.add_msg("user", i)
        _ = ollama.stream_response()

if __name__ == "__main__":
    main()
