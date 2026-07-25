import os

DATASET_FILE = "master_dataset.txt"

def append_to_dataset(text):
    with open(DATASET_FILE, "a", encoding="utf-8") as f:
        f.write(text.strip() + "\n\n")
    print(f"Data berhasil ditambahkan ke {DATASET_FILE}")

if __name__ == "__main__":
    print("Masukkan teks berkualitas tinggi untuk ditambahkan ke dataset:")
    print("(Ketik 'QUIT' di baris baru untuk selesai)")
    
    user_input = ""
    while True:
        line = input()
        if line.strip() == "QUIT":
            break
        user_input += line + "\n"
    
    if user_input.strip():
        append_to_dataset(user_input)
