import data as d1
import sys

def generate_and_save_adversarial_texts(input_text, N, file_path):
    """
    Generates adversarial texts based on an input text, saves them to a file,
    and returns the generated texts.

    The process involves 
    1. Split the input text into individual words.
    2. Generate N adversarial texts for each word separately.
    3. Combine the generated texts for individual words into complete phrases.
    4. Generate N adversarial texts for the entire input text.
    5. Combine the results from steps 3 and 4.
    6. Save the combined results to the specified file.

    Args:
    input_text (str): The input text to generate adversarial texts for.
    N (int): The number of adversarial texts to generate for each word and the whole input text.
    file_path (str): The path to the file where the generated adversarial texts will be saved.

    Returns:
    list: A list of generated adversarial texts.
    """
    # Set flags for generating adversarial texts
    include_partial_phrase = 0
    include_input_words = 0

    # Generate adversarial text for each word in the input text separately
    output = []
    for word in input_text.split():
        adversarial_texts = d1.generate_adversarial_texts(word, N, include_partial_phrase, include_input_words)
        output.append(adversarial_texts)

    # Combine the output
    combined_output = []
    for i in range(N):
        combined_text = " ".join([output[k][i] for k in range(len(output))])
        combined_output.append(combined_text)

    # Generate adversarial text for the whole input text
    adversarial_texts_whole = d1.generate_adversarial_texts(input_text, N, include_partial_phrase, include_input_words)

    # Combine the results
    combined_output.extend(adversarial_texts_whole)

    # Save the output to a file
    with open(file_path, 'w') as file:
        for line in combined_output:
            file.write(f"{line}\n")

    return combined_output

def load_adversarial_texts(file_path):
    """
    Loads adversarial texts from a file.

    Args:
    file_path (str): The path to the file containing the adversarial texts.

    Returns:
    list: A list of loaded adversarial texts.
    """
    # Load the output from a file
    with open(file_path, 'r') as file:
        adversarial_texts = [line.strip() for line in file.readlines()]
    return adversarial_texts

if __name__ == "__main__":
    # Example usage as input arguments
    if len(sys.argv) != 4:
        print("Usage: python adversarial_text.py <input_text> <N> <file_path>")
        sys.exit(1)
    
    input_text = sys.argv[1]
    N = int(sys.argv[2])
    file_path = sys.argv[3]

    # Generate and save adversarial texts
    generated_texts = generate_and_save_adversarial_texts(input_text, N, file_path)
    print("Generated texts:")
    for text in generated_texts:
        print(text)

    # Load adversarial texts from file
    loaded_texts = load_adversarial_texts(file_path)
    print("Loaded texts:")
    for text in loaded_texts:
        print(text)

