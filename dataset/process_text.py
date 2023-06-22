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