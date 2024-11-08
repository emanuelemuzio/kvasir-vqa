import ollama
result =  ollama.chat(
    model='llama3.2', 
    messages=[{
        'role' : 'user',
        'content' : 'Test'
}])

print(result['message']['content'])