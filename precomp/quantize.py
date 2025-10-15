import os
import faiss

'''
We only quantize the secondary as we want primary to remain as accurate as possible.
We quantize secondary to make it even faster.
'''

JSTOR_DIR = "out/jstor_secondary"

# Parameters for FAISS Product Quantization
nlist = 100                 # number of coarse clusters
m = 8                       # number of subquantizers
nbits = 8                   # bits per subvector
TRAIN_SAMPLE = 10000        # number of vectors to use for training
BATCH_SIZE = 100000         # number of vectors per batch during add()

def quantize_segment(segment_path):
    index_path = os.path.join(segment_path, "index.faiss")
    if not os.path.exists(index_path):
        print(f"No FAISS index found in {segment_path}, skipping.")
        return

    print(f"Quantizing: {segment_path}")
    index = faiss.read_index(index_path)

    d = index.d
    print(f"  → Dimension: {d}, Vectors: {index.ntotal}")

    quantizer = faiss.IndexFlatL2(d)
    index_pq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

    # Train on sample
    train_count = min(TRAIN_SAMPLE, index.ntotal)
    xb = index.reconstruct_n(0, train_count)
    print(f"  → Training PQ on {train_count} samples...")
    index_pq.train(xb)

    # Add all vectors in batches
    total = index.ntotal
    for i in range(0, total, BATCH_SIZE):
        batch_count = min(BATCH_SIZE, total - i)
        batch = index.reconstruct_n(i, batch_count)
        index_pq.add(batch)
        print(f"  → Added {i + batch_count}/{total} vectors")

    # Save compressed version
    quantized_path = os.path.join(segment_path, "index_quantized.faiss")
    faiss.write_index(index_pq, quantized_path)
    print(f"Saved quantized index: {quantized_path}\n")

def main():
    for folder in os.listdir(JSTOR_DIR):
        if folder.startswith("segment_"):
            quantize_segment(os.path.join(JSTOR_DIR, folder))

if __name__ == "__main__":
    main()
