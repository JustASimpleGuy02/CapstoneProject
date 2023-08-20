import string
import re

def remove_punctuation(text: str):
    """
    Remove punctuation from string

    Args:
        text (str): input string

    Returns:
        str: string with punctuation removed
    """
    
    result = text.translate(str.maketrans('', '', string.punctuation))
    return result

def remove_unwant_spaces(text: str):
    """
    Remove unwant spaces from string

    Args:
        text (str): input string

    Returns:
        str: string with unwant spaces removed
    """
    result = text.strip()
    
    result = re.sub('\s+',' ', result)
    return result


def replace_punctuation_with_whitespace(input_string):
    translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    modified_string = input_string.translate(translation_table)
    
    return modified_string
