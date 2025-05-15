import re
import json


def extract_code_from_backticks(text: str) -> str:
    """
    Extracts the code block from a string enclosed in triple backticks.

    Args:
        text (str): The input string containing a code block.

    Returns:
        str: The extracted code block.
    """
    match = re.search(r"```(?:\w+)?\n([\s\S]+?)\n```", text)
    return (
        match.group(1) if match else text
    )  # Return extracted code or the original text if no match


def extract_turns(conversation_text):
    pattern = r"Turn-\d+:\s*(.*?)\s*(?=Turn-\d+:|$)"
    return re.findall(pattern, conversation_text, flags=re.DOTALL)


def extract_test_case(code_str, category="capability", index=0):
    # First, find the beginning of the desired category within testcases.
    testcases_idx = code_str.find("testcases")
    if testcases_idx == -1:
        raise ValueError("Could not find 'testcases' in the input string.")

    # Find the desired category key (e.g., "capability")
    cat_key = f'"{category}"'
    cat_idx = code_str.find(cat_key, testcases_idx)
    if cat_idx == -1:
        raise ValueError(f"Category {category} not found in testcases.")

    # Find the opening bracket '[' that starts the list for this category.
    bracket_idx = code_str.find("[", cat_idx)
    if bracket_idx == -1:
        raise ValueError(f"Opening bracket for category {category} not found.")

    # Look for the nth tuple in the list.
    current_index = bracket_idx
    tuple_str = None
    for _ in range(index + 1):
        # find the start of the next tuple (marked by an opening parenthesis)
        tuple_start = code_str.find("(", current_index)
        if tuple_start == -1:
            raise ValueError("Not enough test cases in the specified category.")
        tuple_end = find_matching_paren(code_str, tuple_start)
        if tuple_end == -1:
            raise ValueError("Could not find matching closing parenthesis for tuple.")
        if _ == index:
            tuple_str = code_str[tuple_start : tuple_end + 1]
            break
        current_index = tuple_end + 1

    if tuple_str is None:
        raise ValueError("Test case tuple not found.")

    # Now split the tuple into its two parts (input and output).
    input_part, output_part = split_tuple(tuple_str)

    # Format the output string.
    result = (
        "{'input': \n    "
        + input_part.strip()
        + ", \n'output': \n    "
        + output_part.strip()
        + "\n}"
    )
    return result


def find_matching_paren(s, start_index):
    """
    Given a string s and an index (start_index) that must be the position
    of an opening parenthesis '(', return the index of its matching ')'.
    This simple parser also skips over string literals.
    """
    if s[start_index] != "(":
        raise ValueError("The character at start_index is not an opening parenthesis.")
    stack = 1
    i = start_index + 1
    while i < len(s):
        c = s[i]
        if c == "(":
            stack += 1
        elif c == ")":
            stack -= 1
            if stack == 0:
                return i
        elif c in ("'", '"'):
            # Skip over string literal: find the matching quote.
            quote_char = c
            i += 1
            while i < len(s) and s[i] != quote_char:
                if s[i] == "\\":  # skip escaped characters
                    i += 1
                i += 1
        i += 1
    return -1  # no matching parenthesis found


def split_tuple(tuple_str):
    """
    Given a tuple string of the form
        (<input_part>, <output_part>)
    (including the outer parentheses) this function returns a tuple of two strings.
    It does a simple parse that looks for the top-level comma while skipping nested
    parentheses, brackets, braces, and string literals.
    """
    # Remove the outer parentheses.
    inner = tuple_str[1:-1]

    level_paren = 0
    level_brace = 0
    level_bracket = 0
    comma_index = None
    i = 0
    while i < len(inner):
        c = inner[i]
        if c in ("'", '"'):
            # Skip string literal.
            quote_char = c
            i += 1
            while i < len(inner) and inner[i] != quote_char:
                if inner[i] == "\\":
                    i += 1
                i += 1
        elif c == "(":
            level_paren += 1
        elif c == ")":
            level_paren -= 1
        elif c == "{":
            level_brace += 1
        elif c == "}":
            level_brace -= 1
        elif c == "[":
            level_bracket += 1
        elif c == "]":
            level_bracket -= 1
        elif c == "," and level_paren == 0 and level_brace == 0 and level_bracket == 0:
            comma_index = i
            break
        i += 1

    if comma_index is None:
        raise ValueError("Could not split tuple: comma not found at top level.")

    part1 = inner[:comma_index]
    part2 = inner[comma_index + 1 :]
    return part1, part2


def transform_variable_to_json(testcases_str):
    # Use regex to extract the dictionary content
    match = re.search(r"testcases\s*=\s*(\{.*\})", testcases_str, re.DOTALL)
    if not match:
        raise ValueError("Could not extract testcases dictionary from string")

    data_str = match.group(1)

    # Safely evaluate the extracted dictionary
    testcases = eval(data_str, {"ValueError": ValueError})

    # Extracting the components
    input_data, output_data = testcases["capability"][0]

    # Structuring the final dictionary
    formatted_data = {"input": input_data, "output": output_data}

    # Converting to formatted JSON string
    return json.dumps(formatted_data, indent=2)
