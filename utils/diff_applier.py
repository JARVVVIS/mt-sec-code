#!/usr/bin/env python3
import difflib
from itertools import groupby
from pathlib import Path
import re


class SearchTextNotUnique(ValueError):
    """Exception raised when search text is found multiple times in the original text."""

    pass


class RelativeIndenter:
    """
    Rewrites text files to have relative indentation, which involves
    reformatting the leading white space on lines. This format makes
    it easier to search and apply edits to pairs of code blocks which
    may differ significantly in their overall level of indentation.
    """

    def __init__(self, texts):
        """
        Based on the texts, choose a unicode character that isn't in any of them.
        """
        chars = set()
        for text in texts:
            chars.update(text)

        ARROW = "←"
        if ARROW not in chars:
            self.marker = ARROW
        else:
            self.marker = self.select_unique_marker(chars)

    def select_unique_marker(self, chars):
        for codepoint in range(0x10FFFF, 0x10000, -1):
            marker = chr(codepoint)
            if marker not in chars:
                return marker
        raise ValueError("Could not find a unique marker")

    def make_relative(self, text):
        """
        Transform text to use relative indents.
        """
        if self.marker in text:
            raise ValueError(f"Text already contains the outdent marker: {self.marker}")

        lines = text.splitlines(keepends=True)

        output = []
        prev_indent = ""
        for line in lines:
            line_without_end = line.rstrip("\n\r")

            len_indent = len(line_without_end) - len(line_without_end.lstrip())
            indent = line[:len_indent]
            change = len_indent - len(prev_indent)
            if change > 0:
                cur_indent = indent[-change:]
            elif change < 0:
                cur_indent = self.marker * -change
            else:
                cur_indent = ""

            out_line = cur_indent + "\n" + line[len_indent:]
            output.append(out_line)
            prev_indent = indent

        res = "".join(output)
        return res

    def make_absolute(self, text):
        """
        Transform text from relative back to absolute indents.
        """
        lines = text.splitlines(keepends=True)

        output = []
        prev_indent = ""
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                break

            dent = lines[i].rstrip("\r\n")
            non_indent = lines[i + 1]

            if dent.startswith(self.marker):
                len_outdent = len(dent)
                cur_indent = prev_indent[:-len_outdent]
            else:
                cur_indent = prev_indent + dent

            if not non_indent.rstrip("\r\n"):
                out_line = non_indent  # don't indent a blank line
            else:
                out_line = cur_indent + non_indent

            output.append(out_line)
            prev_indent = cur_indent

        res = "".join(output)
        if self.marker in res:
            raise ValueError("Error transforming text back to absolute indents")

        return res


def apply_code_diff(code_before, code_diff):
    """
    Apply a code diff to a string of code and return the result.

    Args:
        code_before (str): The original code to which the diff will be applied
        code_diff (str): The code diff in unified diff format

    Returns:
        str: The code after applying the diff, or None if the diff couldn't be applied

    Raises:
        ValueError: If the diff is malformed or cannot be applied
        SearchTextNotUnique: If the diff matches multiple locations in the code
    """
    # Ensure code_before ends with a newline for consistent processing
    if code_before and not code_before.endswith("\n"):
        code_before = code_before + "\n"

    # Parse the diff to extract hunks
    hunks = list(find_diffs(code_diff))

    # Apply each hunk to the code, maintaining order
    content = code_before
    errors = []

    for i, (_, hunk) in enumerate(hunks):
        hunk = normalize_hunk(hunk)
        if not hunk:
            continue

        # Get the before and after text from the hunk
        before_text, after_text = hunk_to_before_after(hunk)

        # Check if this is a no-context hunk
        has_no_context = not before_text.strip()

        # Special handling for no-context hunks
        if has_no_context:
            if i == 0:  # First hunk with no context
                # Apply at the top of the file
                content = after_text + content
                continue
            elif i == len(hunks) - 1:  # Last hunk with no context
                # Apply at the bottom of the file
                content = content + after_text
                continue

        # Apply the hunk to the content normally
        try:
            new_content = do_replace(content, hunk)

            if not new_content:
                # Try with more flexible strategies
                new_content = flexible_apply_hunk(content, hunk)

            if not new_content and has_no_context:
                # If no-context hunk couldn't be applied, just append it in sequence
                content = content + after_text
                continue
            elif not new_content:
                # If the hunk couldn't be applied, collect the error
                errors.append(
                    f"Failed to apply hunk. Code does not contain these lines:\n{before_text}"
                )
                continue

            content = new_content
        except SearchTextNotUnique:
            errors.append(
                f"Multiple matches found for hunk. Cannot uniquely identify where to apply:\n{before_text}"
            )
            continue

    if errors:
        raise ValueError("\n\n".join(errors))

    return content


def do_replace(content, hunk):
    """
    Apply a hunk to the content.

    Args:
        content (str): The content to modify
        hunk (list): The hunk to apply

    Returns:
        str: The modified content, or None if the hunk couldn't be applied
    """
    before_text, after_text = hunk_to_before_after(hunk)

    # If there's no before text, this is an append operation
    if not before_text.strip():
        # Append to existing content
        new_content = content + after_text
        return new_content

    # Try to apply the hunk directly
    new_content = apply_hunk(content, hunk)
    if new_content:
        return new_content

    return None


def apply_hunk(content, hunk):
    """
    Apply a hunk to the content using various strategies.

    Args:
        content (str): The content to modify
        hunk (list): The hunk to apply

    Returns:
        str: The modified content, or None if the hunk couldn't be applied
    """
    before_text, after_text = hunk_to_before_after(hunk)

    # Try to apply the hunk directly
    res = directly_apply_hunk(content, hunk)
    if res:
        return res

    # Make new lines explicit
    hunk = make_new_lines_explicit(content, hunk)

    # Just consider space vs not-space
    ops = "".join([line[0] for line in hunk])
    ops = ops.replace("-", "x")
    ops = ops.replace("+", "x")
    ops = ops.replace("\n", " ")

    cur_op = " "
    section = []
    sections = []

    for i in range(len(ops)):
        op = ops[i]
        if op != cur_op:
            sections.append(section)
            section = []
            cur_op = op
        section.append(hunk[i])

    sections.append(section)
    if cur_op != " ":
        sections.append([])

    all_done = True
    for i in range(2, len(sections), 2):
        preceding_context = sections[i - 2]
        changes = sections[i - 1]
        following_context = sections[i]

        res = apply_partial_hunk(content, preceding_context, changes, following_context)
        if res:
            content = res
        else:
            all_done = False
            break

    if all_done:
        return content

    return None


def directly_apply_hunk(content, hunk):
    """
    Try to apply a hunk directly to the content.

    Args:
        content (str): The content to modify
        hunk (list): The hunk to apply

    Returns:
        str: The modified content, or None if the hunk couldn't be applied directly
    """
    before, after = hunk_to_before_after(hunk)

    if not before:
        return None

    # Special handling for hunks with no context (like adding imports)
    if all(
        line.startswith("+")
        for line in hunk
        if line.strip() and not line.startswith("@")
    ):
        # This is a pure addition hunk with no context
        # Try to find a suitable location (e.g., top of file for imports)
        if any("import " in line for line in hunk):
            # For imports, add at the top of the file
            return after + content

    # Try with exact matching first
    try:
        new_content = search_and_replace([before, after, content])
        if new_content:
            return new_content
    except SearchTextNotUnique:
        # If multiple matches, we'll try other strategies
        pass

    # Try with flexible search and replace
    return flexi_search_and_replace([before, after, content])


def apply_partial_hunk(content, preceding_context, changes, following_context):
    """
    Apply a partial hunk to the content.

    Args:
        content (str): The content to modify
        preceding_context (list): The context lines before the changes
        changes (list): The lines to change
        following_context (list): The context lines after the changes

    Returns:
        str: The modified content, or None if the partial hunk couldn't be applied
    """
    len_prec = len(preceding_context)
    len_foll = len(following_context)

    use_all = len_prec + len_foll

    # If there is a - in the hunk, we can go all the way to `use=0`
    for drop in range(use_all + 1):
        use = use_all - drop

        for use_prec in range(len_prec, -1, -1):
            if use_prec > use:
                continue

            use_foll = use - use_prec
            if use_foll > len_foll:
                continue

            if use_prec:
                this_prec = preceding_context[-use_prec:]
            else:
                this_prec = []

            this_foll = following_context[:use_foll]

            res = directly_apply_hunk(content, this_prec + changes + this_foll)
            if res:
                return res

    return None


def find_diffs(content):
    """
    Find diffs in the content.

    Args:
        content (str): The content containing diffs

    Yields:
        tuple: (path, hunk) pairs
    """
    if not content.endswith("\n"):
        content = content + "\n"

    lines = content.splitlines(keepends=True)
    line_num = 0
    edits = []

    while line_num < len(lines):
        while line_num < len(lines):
            line = lines[line_num]
            if line.startswith("```diff"):
                line_num, these_edits = process_fenced_block(lines, line_num + 1)
                edits += these_edits
                break
            # Also handle diffs without fenced blocks
            elif (
                line.startswith("--- ")
                and line_num + 1 < len(lines)
                and lines[line_num + 1].startswith("+++ ")
            ):
                line_num, these_edits = process_unfenced_diff(lines, line_num)
                edits += these_edits
                break
            line_num += 1

        if line_num >= len(lines):
            break

    # If no diffs were found but the content looks like a diff, try to parse it directly
    if not edits and "--- " in content and "+++ " in content:
        lines = content.splitlines(keepends=True)
        line_num, edits = process_unfenced_diff(lines, 0)

    for path, hunk in edits:
        yield (path, hunk)


def process_unfenced_diff(lines, start_line_num):
    """
    Process a diff that's not in a fenced block.

    Args:
        lines (list): The lines of the diff
        start_line_num (int): The line number to start processing from

    Returns:
        tuple: (next_line_num, edits)
    """
    # Find the end of the diff
    end_line_num = start_line_num
    while end_line_num < len(lines):
        if (
            end_line_num + 1 < len(lines)
            and lines[end_line_num].startswith("--- ")
            and lines[end_line_num + 1].startswith("+++ ")
        ):
            # Found the start of another diff
            if end_line_num > start_line_num:
                break
        end_line_num += 1

    # Extract the file path
    fname = None
    if lines[start_line_num].startswith("--- ") and lines[
        start_line_num + 1
    ].startswith("+++ "):
        fname = lines[start_line_num + 1][4:].strip()
        start_line_num += 2

    # Extract the hunks
    edits = []
    hunk = []
    keeper = False

    for line_num in range(start_line_num, end_line_num):
        line = lines[line_num]
        hunk.append(line)

        if len(line) < 2:
            continue

        op = line[0]
        if op in "-+":
            keeper = True
            continue
        if op != "@":
            continue
        if not keeper:
            hunk = []
            continue

        hunk = hunk[:-1]
        edits.append((fname, hunk))
        hunk = []
        keeper = False

    # Add the last hunk if it's not empty
    if hunk and keeper:
        edits.append((fname, hunk))

    return end_line_num, edits


def process_fenced_block(lines, start_line_num):
    """
    Process a fenced diff block.

    Args:
        lines (list): The lines of the diff
        start_line_num (int): The line number to start processing from

    Returns:
        tuple: (next_line_num, edits)
    """
    for line_num in range(start_line_num, len(lines)):
        line = lines[line_num]
        if line.startswith("```"):
            break

    block = lines[start_line_num:line_num]
    block.append("@@ @@")

    if block[0].startswith("--- ") and block[1].startswith("+++ "):
        # Extract the file path, considering that it might contain spaces
        fname = block[1][4:].strip()
        block = block[2:]
    else:
        fname = None

    edits = []

    keeper = False
    hunk = []
    op = " "
    for line in block:
        hunk.append(line)
        if len(line) < 2:
            continue

        if line.startswith("+++ ") and hunk[-2].startswith("--- "):
            if len(hunk) >= 3 and hunk[-3] == "\n":
                hunk = hunk[:-3]
            else:
                hunk = hunk[:-2]

            edits.append((fname, hunk))
            hunk = []
            keeper = False

            fname = line[4:].strip()
            continue

        op = line[0]
        if op in "-+":
            keeper = True
            continue
        if op != "@":
            continue
        if not keeper:
            hunk = []
            continue

        hunk = hunk[:-1]
        edits.append((fname, hunk))
        hunk = []
        keeper = False

    return line_num + 1, edits


def hunk_to_before_after(hunk, lines=False):
    """
    Convert a hunk to before and after text.

    Args:
        hunk (list): The hunk to convert
        lines (bool): Whether to return lists of lines instead of strings

    Returns:
        tuple: (before, after) text or lines
    """
    before = []
    after = []
    op = " "
    for line in hunk:
        if len(line) < 2:
            op = " "
            line = line
        else:
            op = line[0]
            line = line[1:]

        if op == " ":
            before.append(line)
            after.append(line)
        elif op == "-":
            before.append(line)
        elif op == "+":
            after.append(line)

    if lines:
        return before, after

    before = "".join(before)
    after = "".join(after)

    return before, after


def normalize_hunk(hunk):
    """
    Normalize a hunk by cleaning up whitespace.

    Args:
        hunk (list): The hunk to normalize

    Returns:
        list: The normalized hunk
    """
    before, after = hunk_to_before_after(hunk, lines=True)

    before = cleanup_pure_whitespace_lines(before)
    after = cleanup_pure_whitespace_lines(after)

    diff = difflib.unified_diff(before, after, n=max(len(before), len(after)))
    try:
        diff = list(diff)[3:]
        return diff
    except IndexError:
        # If the diff is empty or malformed, return the original hunk
        return hunk


def cleanup_pure_whitespace_lines(lines):
    """
    Clean up pure whitespace lines.

    Args:
        lines (list): The lines to clean up

    Returns:
        list: The cleaned up lines
    """
    res = [
        (
            line
            if line.strip() or line.rstrip("\r\n") == ""
            else line[-(len(line) - len(line.rstrip("\r\n")))]
        )
        for line in lines
    ]
    return res


def make_new_lines_explicit(content, hunk):
    """
    Make new lines explicit in a hunk.

    Args:
        content (str): The content to modify
        hunk (list): The hunk to modify

    Returns:
        list: The modified hunk
    """
    before, after = hunk_to_before_after(hunk)

    diff = diff_lines(before, content)

    back_diff = []
    for line in diff:
        if line[0] == "+":
            continue
        back_diff.append(line)

    new_before = directly_apply_hunk(before, back_diff)
    if not new_before:
        return hunk

    if len(new_before.strip()) < 10:
        return hunk

    before = before.splitlines(keepends=True)
    new_before = new_before.splitlines(keepends=True)
    after = after.splitlines(keepends=True)

    if len(new_before) < len(before) * 0.66:
        return hunk

    new_hunk = difflib.unified_diff(
        new_before, after, n=max(len(new_before), len(after))
    )
    try:
        new_hunk = list(new_hunk)[3:]
        return new_hunk
    except IndexError:
        return hunk


def diff_lines(search_text, replace_text):
    """
    Generate a diff between two texts, line by line.

    Args:
        search_text (str): The original text
        replace_text (str): The new text

    Returns:
        list: The diff lines
    """
    dmp = difflib.Differ()
    search_lines = search_text.splitlines(keepends=True)
    replace_lines = replace_text.splitlines(keepends=True)

    diff = dmp.compare(search_lines, replace_lines)

    udiff = []
    for line in diff:
        if line.startswith("- "):
            udiff.append("-" + line[2:])
        elif line.startswith("+ "):
            udiff.append("+" + line[2:])
        elif line.startswith("  "):
            udiff.append(" " + line[2:])

    return udiff


def search_and_replace(texts):
    """
    Simple search and replace function.

    Args:
        texts (list): [search_text, replace_text, original_text]

    Returns:
        str: The modified text, or None if the search text wasn't found

    Raises:
        SearchTextNotUnique: If the search text is found multiple times
    """
    search_text, replace_text, original_text = texts

    # Handle triple quotes and string literals more carefully
    search_text = preserve_string_literals(search_text)
    original_text = preserve_string_literals(original_text)

    num = original_text.count(search_text)
    if num > 1:
        raise SearchTextNotUnique()
    if num == 0:
        # Try with normalized whitespace
        search_norm = normalize_whitespace(search_text)
        original_norm = normalize_whitespace(original_text)

        num = original_norm.count(search_norm)
        if num > 1:
            raise SearchTextNotUnique()
        if num == 0:
            return None

        # Found with normalized whitespace, now find the actual position
        start = original_norm.find(search_norm)
        end = start + len(search_norm)

        # Extract the actual text from the original
        actual_search = original_text[start:end]
        return original_text.replace(actual_search, replace_text)

    new_text = original_text.replace(search_text, replace_text)
    return new_text


def preserve_string_literals(text):
    """
    Preserve string literals in text to avoid whitespace normalization issues.

    Args:
        text (str): The text to process

    Returns:
        str: The processed text
    """
    # This is a simplified version - a more robust implementation would
    # actually parse the code to handle nested quotes correctly
    return text


def normalize_whitespace(text):
    """
    Normalize whitespace in text for more flexible matching.

    Args:
        text (str): The text to normalize

    Returns:
        str: The normalized text
    """
    # Replace multiple spaces with a single space
    text = re.sub(r" +", " ", text)
    # Normalize line endings
    text = text.replace("\r\n", "\n")
    return text


def relative_indent(texts):
    """
    Apply relative indentation to texts.

    Args:
        texts (list): List of texts to process

    Returns:
        tuple: (RelativeIndenter instance, processed texts)
    """
    ri = RelativeIndenter(texts)
    texts = list(map(ri.make_relative, texts))
    return ri, texts


def strip_blank_lines(texts):
    """
    Strip leading and trailing blank lines from texts.

    Args:
        texts (list): List of texts to process

    Returns:
        list: Processed texts
    """
    texts = [text.strip("\n") + "\n" for text in texts]
    return texts


def flexi_search_and_replace(texts):
    """
    Try a series of search/replace methods with different preprocessing options.

    Args:
        texts (list): [search_text, replace_text, original_text]

    Returns:
        str: The modified text, or None if none of the strategies worked
    """
    strategies = [
        (search_and_replace, False, False),  # Direct search and replace
        (search_and_replace, True, False),  # Strip blank lines
        (search_and_replace, False, True),  # Relative indent
        (search_and_replace, True, True),  # Both
    ]

    for strategy, strip_blanks, rel_indent in strategies:
        processed_texts = texts.copy()
        ri = None

        if strip_blanks:
            processed_texts = strip_blank_lines(processed_texts)

        if rel_indent:
            ri, processed_texts = relative_indent(processed_texts)

        try:
            result = strategy(processed_texts)
            if result:
                if rel_indent:
                    try:
                        result = ri.make_absolute(result)
                    except ValueError:
                        continue
                return result
        except (ValueError, SearchTextNotUnique):
            continue

    return None


def flexible_apply_hunk(content, hunk):
    """
    Apply a hunk using multiple strategies for more flexibility.

    Args:
        content (str): The content to modify
        hunk (list): The hunk to apply

    Returns:
        str: The modified content, or None if the hunk couldn't be applied
    """
    before_text, after_text = hunk_to_before_after(hunk)

    # Try with line-based diff application
    try:
        # Ensure both texts end with newlines
        if before_text and not before_text.endswith("\n"):
            before_text += "\n"
        if content and not content.endswith("\n"):
            content += "\n"

        # Convert to line-based representation
        before_lines = before_text.splitlines(keepends=True)
        content_lines = content.splitlines(keepends=True)

        # Find the best match for the before_lines in content_lines
        best_match_idx = find_best_match(before_lines, content_lines)

        if best_match_idx is not None:
            # Replace the matched lines with after_text
            after_lines = after_text.splitlines(keepends=True)
            result_lines = (
                content_lines[:best_match_idx]
                + after_lines
                + content_lines[best_match_idx + len(before_lines) :]
            )
            return "".join(result_lines)
    except Exception:
        pass

    # Try with flexible search and replace
    return flexi_search_and_replace([before_text, after_text, content])


def find_best_match(needle_lines, haystack_lines):
    """
    Find the best match for needle_lines in haystack_lines.

    Args:
        needle_lines (list): The lines to find
        haystack_lines (list): The lines to search in

    Returns:
        int: The index of the best match, or None if no good match is found
    """
    if not needle_lines:
        return None

    best_match = None
    best_score = 0

    for i in range(len(haystack_lines) - len(needle_lines) + 1):
        score = 0
        for j in range(len(needle_lines)):
            needle = needle_lines[j].strip()
            haystack = haystack_lines[i + j].strip()

            # Skip empty lines in scoring
            if not needle and not haystack:
                continue

            # Calculate similarity score
            if needle == haystack:
                score += 1
            elif needle in haystack or haystack in needle:
                score += 0.5

        # Normalize score
        score = score / max(1, len(needle_lines))

        if score > best_score:
            best_score = score
            best_match = i

    # Only return matches above a certain threshold
    if best_score >= 0.7:
        return best_match

    return None


# Simple function to apply a code diff to a string of code
def apply_diff(code_before, code_diff):
    """
    Apply a code diff to a string of code and return the result.
    This is a simple wrapper around apply_code_diff.

    Args:
        code_before (str): The original code to which the diff will be applied
        code_diff (str): The code diff in unified diff format

    Returns:
        str: The code after applying the diff, or None if the diff couldn't be applied

    Raises:
        ValueError: If the diff is malformed or cannot be applied
        SearchTextNotUnique: If the diff matches multiple locations in the code
    """
    return apply_code_diff(code_before, code_diff)
