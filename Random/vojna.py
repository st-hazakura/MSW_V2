import PyPDF2  # PyPDF2 pro extrakci textu z PDF souboru
import random

pdf_path = r'C:\Users\Home\Documents\GitHub\MSW_V2\Tolstoy_Voina_i_mir.pdf'
with open(pdf_path, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)  # Vytvoření objektu pro práci s PDF

    characters = []  # Každá stránka PDF je extrahována jako text a znaky tohoto textu jsou přidávány do pole characters
    for page in pdf_reader.pages:
        text = page.extract_text()
        characters.extend(list(text))

# Proces přiřazení indexu každému znaku v mapě
character_map = {}
for index, char in enumerate(set(characters)):  # enumerate - prochází unikátní znaky získané z set(characters)
    character_map[char] = index


def random_number_generator():  # Generování náhodných čísel a provádění náhodných matematických operací s nimi
    while True:
        num1 = random.randint(0, len(character_map) - 1)
        num2 = random.randint(0, len(character_map) - 1)
        operation = random.choice(['+', '-', '*', '/'])
        
        if operation == '+':
            result = num1 + num2
        elif operation == '-':
            result = num1 - num2
        elif operation == '*':
            result = num1 * num2
        elif operation == '/':
            # Ověření dělení nulou
            if num2 != 0:
                result = num1 / num2
            else:
                result = float('inf')  # Infinity

        yield result  # Používáno pro vrácení výsledků matematických operací jako posloupnost náhodných čísel.
        # Každé volání next(rng) vrátí další náhodné číslo v závislosti na vybrané matematické operaci.


rng = random_number_generator()  # Vytvoření instance generátoru náhodných čísel
for _ in range(10):  # Proměnná _ se používá pro označení dočasné proměnné, která se v těle smyčky nepoužívá. Vygenerujeme 10 pseudonáhodných čísel.
    print(next(rng))

# print(character_map)
