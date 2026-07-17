from tqdm import tqdm, trange
from time import sleep

# for i in tqdm(range(100000000)):
#     pass

# equivalent
for i in trange(10000000):
    pass

# exemple avec liste ou chaque element est un stade de progres
text = ""
for char in tqdm(["a", "b", "c", "d"]):
    sleep(0.25)
    text = text + char

# Fonctionne aussi si dans variable
pbar = tqdm(["a", "b", "c", "d"])
for char in pbar:
    sleep(0.25)
    pbar.set_description("Processing %s" % char)
    # pour nommer la barre
