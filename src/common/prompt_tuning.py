import ollama

class PromptTuning:
    
    '''
    Chain of thought mechanism for VQA enhancing
    
    ----------
    Parameters
        ollama_model: str
            Name of the ollama powered model being used. 
            Make sure it exists by running the following script:
                - ollama create {ollama_model} -f ./Modelfile
    ----------
    '''
    
    def __init__(self, ollama_model=None): 
        self.model = ollama_model

    def generate(self, question):
        
        output = []
        
        if isinstance(question, str):
            output.append(ollama.chat(
                model=self.model, 
                messages=[{
                    'role' : 'user',
                    'content' : question
                }])['message']['content'])
        elif isinstance(question, list) or isinstance(question, tuple):
            for q in question:
                output.append(ollama.chat(
                    model=self.model, 
                    messages=[{
                        'role' : 'user',
                        'content' : q
                    }])['message']['content'])
        return output 