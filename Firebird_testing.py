import os

files = os.listdir("./data")

for file in files:
    if file.endswith(".h5"):
        os.remove(os.path.join("./data", file))

