import os
import re

def extract_keyword(answer):
    """
    Extracts the main keyword from a given answer string by removing common leading phrases.
    Returns the cleaned keyword.
    """
    patterns = [
        r'^(organ being operated is|action done by [a-z ]+ is|action done by [a-z ]+ are|the tools operating are|[a-z ]+ is located at) ',
        r'^(.*? is|.*? are|.*? at) '
    ]
    for pat in patterns:
        new_answer = re.sub(pat, '', answer, flags=re.IGNORECASE)
        if new_answer != answer:
            return new_answer.strip()
    return answer.strip()  # fallback

def process_file(input_path, output_path):
    """
    Reads a QA text file, extracts keywords from all answers,
    and writes the processed QA pairs to a new file.
    """
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if '|' not in line:
                continue
            q, a = line.split('|', 1)
            a_kw = extract_keyword(a)
            fout.write(f"{q}|{a_kw}\n")

def main():
    """
    Iterates over all sequence folders (except 8 and 13),
    processes all QA files starting with "frame" in the Sentence folder,
    and writes keyword-extracted versions to the Keyword folder.
    """
    for i in range(1,17):
        if i in [8, 13]:
            continue
        root = f"/SAN/medic/Kvasir/EndoVis-18-VQA/dataset/seq_{i}/vqa" # substitute with proper data location
        in_folder = os.path.join(root, "Sentence")
        out_folder = os.path.join(root, "Keyword")
        os.makedirs(out_folder, exist_ok=True)

        for fname in os.listdir(in_folder):
            if fname.startswith("frame") and fname.endswith("_QA.txt"):
                in_path = os.path.join(in_folder, fname)
                out_path = os.path.join(out_folder, fname)
                process_file(in_path, out_path)
                print(f"Processed {fname}")

if __name__ == "__main__":
    main()