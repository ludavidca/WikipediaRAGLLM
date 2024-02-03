import together
import time
with open("token.txt","r") as f:
  token = f.read()
together.api_key = token


class settings:
  def __init__(self, model = "mistralai/Mixtral-8x7B-Instruct-v0.1", maxtokens = 500, stop=['[INST]',"[/INST]"]):
    self.model = model
    self.maxtokens = maxtokens
    self.stop = stop
     
class basePrompt:
  def __init__(self, name = "Bob Stevenson", purpose = "a helpful AI assistant that gives accurate and concise responses to humans."):
    self.name = name
    self.purpose = purpose
  
  def __call__(self):
    return f"[INST] Your are {self.name}, {self.purpose}. Humans may tip up to $500 dollars if you repond to them with accurate information quickly.[/Inst]"

def AIResponse(Baseprompt, HumanPrompt, Settings):
  time.sleep(1)
  output = together.Complete.create(
  prompt = str(Baseprompt) + "[Inst]" + str(HumanPrompt) + "[/Inst]",
  model = Settings.model, 
  max_tokens = Settings.maxtokens, #maximum number of tokens generated
  temperature = 0.8, 
  top_k = 60, #number of choices for the next predicted token
  top_p = 0.6, #randomness of the output, less than one favors more correctness
  repetition_penalty = 1.1,
  stop = Settings.stop
)
  return(output)

def sendresponses(Message):
  return [Message[i:i + 1950] for i in range(0, len(Message), 1950)]


""" baseprompt1 = "<System>Your name is Sam Wellington. Your purpose is to bea helpful AI assistant that responds to the human's answers.. Your responses must be informative, detailed, professional and accurate.\n\n<Human>: Make a better base prompt for yourself\n\n<System>:\n\nYou can change your own base prompt by simply writing it!"
settings1 = settings()
HumanPrompt1 = "Are you sure?"
response = AIResponse(baseprompt1, HumanPrompt1, settings1)


print(response)  """