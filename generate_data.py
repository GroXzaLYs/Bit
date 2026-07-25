import random

# Dataset generator yang lebih canggih & kaya konten
languages = {
    "en": {
        "greeting": ["Hello, how are you?", "Hi there!", "Good morning, everyone."],
        "tech": ["Python is a versatile language.", "Machine learning is the future of computing.", "Deep learning models require lots of data."],
        "science": ["The speed of light is constant.", "Water is essential for life.", "Gravity keeps planets in orbit."],
        "story": ["Once upon a time in a distant land...", "The brave knight journeyed to the mountain.", "The stars shone brightly in the night sky."]
    },
    "id": {
        "greeting": ["Halo, apa kabar?", "Hai teman-teman!", "Selamat pagi semuanya."],
        "tech": ["Python adalah bahasa yang fleksibel.", "Machine learning adalah masa depan komputasi.", "Model deep learning butuh banyak data."],
        "science": ["Kecepatan cahaya itu konstan.", "Air sangat penting untuk kehidupan.", "Gravitasi menjaga planet tetap di orbit."],
        "story": ["Pada zaman dahulu di negeri yang jauh...", "Ksatria pemberani itu pergi ke gunung.", "Bintang bersinar terang di langit malam."]
    },
    "ja": {
        "greeting": ["こんにちは、元気ですか？", "やあ！", "皆さん、おはようございます。"],
        "tech": ["Pythonは多用途な言語です。", "機械学習はコンピューティングの未来です。", "ディープラーニングモデルには多くのデータが必要です。"],
        "science": ["光速は一定です。", "水は生命にとって不可欠です。", "重力が惑星を軌道に保ちます。"],
        "story": ["昔々、遠い国に...", "勇敢な騎士は山へ旅立ちました。", "夜空には星が明るく輝いていました。"]
    }
}

def generate_large_dataset(filename="data_multilingual_massive.txt", num_lines=50000):
    print(f"Generating {num_lines} lines of data...")
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(num_lines):
            lang = random.choice(list(languages.keys()))
            topic = random.choice(list(languages[lang].keys()))
            sentence = random.choice(languages[lang][topic])
            
            # Tambahkan variasi agar tidak terlalu repetitif
            if random.random() > 0.5:
                sentence = sentence + " " + random.choice(languages[lang]["greeting"])
                
            f.write(sentence + "\n")
    print(f"Dataset '{filename}' berhasil dibuat dengan {num_lines} baris.")

if __name__ == "__main__":
    generate_large_dataset()
