import modal

app = modal.App("i-m-modaling")

@app.function()
def square(x):
    print("This code is running on a remote server of modal")
    return x ** 2

@app.local_entrypoint()
def main():
    print("this square is", square.remote(42))