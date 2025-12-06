import sys

class Tee:
    """
    Redirects stdout to both the console and a file simultaneously.
    Useful for logging all print statements to a file while still showing them on the screen.
    """

    def __init__(self, filename):
        # Open the file for writing and keep a reference to the original stdout
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        # Write the output to both the console and the file
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        # Flush both the console and the file buffers
        self.stdout.flush()
        self.file.flush()
