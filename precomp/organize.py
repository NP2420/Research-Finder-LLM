import os

'''
Moving folders to appropriate places, discarding unimportant files, and renaming
'''

JSTOR_DIR = "out"
SECONDARY = "jstor_secondary"
OUT = "../resources/jstor"

def organize():
    if len(os.listdir(JSTOR_DIR)) == 0:
        print("Empty out folder.")
        return

    for folder in os.listdir(JSTOR_DIR):
        path = os.path.join(JSTOR_DIR, folder)

        #Secondary is the quantized folder
        if folder == SECONDARY:
            for segment in os.listdir(path):
                index = os.path.join(path, segment, "index.faiss")
                quant = os.path.join(path, segment, "index_quantized.faiss")

                if (os.path.exists(quant)):
                    os.remove(index)
                    os.rename(quant, index)

        out = os.path.join(OUT, folder)
        os.rename(path, out)
        print(f"Moved {path} to {out}.")

def main():
    organize()

if __name__ == "__main__":
    main()
