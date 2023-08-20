from process_text import *

text = " Gold and black silk blend floral jacquard trousers from Christian Pellizzari. Color: Metallic. Gender: Female. Material: Viscose/Polyester/Silk/Polyamide."

# text = process_text.remove_punctuation(text)
text = replace_punctuation_with_whitespace(text)
text = remove_unwant_spaces(text)

print(text)
