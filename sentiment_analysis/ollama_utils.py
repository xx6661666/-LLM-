import subprocess

MODEL_NAME = "deepseek-r1:1.5b"

def query_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", MODEL_NAME],
        input=prompt.encode('utf-8'),
        capture_output=True
    )
    output = result.stdout.decode('utf-8').strip()
    return output