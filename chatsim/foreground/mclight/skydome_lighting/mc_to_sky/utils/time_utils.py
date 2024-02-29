import time

class Timer:
    def __init__(self):
        self.time = time.time()
        print("Start timing ... ")
    
    def print(self, message=""):
        print(f"\n--- {message} using time {time.time() - self.time:3f} ---\n")
        self.time = time.time()