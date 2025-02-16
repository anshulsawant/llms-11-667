from homework import html_to_text, replace_pii, clean_text, heuristic_quality_filter


def test_html_to_text():
    html = b'<!DOCTYPE html><html><head><title>TestPage</title></head><body><h1>Hello,World!</h1><p>Thisisatestpagecontainingalink</a>.</p><ul><li>Item1</li><li>Item2</li><li>Item3</li></ul><p><strong>Boldtext</strong>and<em>italictext</em>.</p></body></html>'
    text = html_to_text(html)

    assert "<" not in text, "< should be stripped out but was not."
    assert ">" not in text, "> should be stripped out but was not."
    assert "DOCTYPE" not in text, "'DOCTYPE' should be stripped out but was not."


def test_replace_pii():
    testcase = "Aria's SSN (123-45-6789) must be updated."
    expected_output = "Aria's SSN (XXX-XX-XXXX) must be updated." 

    assert replace_pii(testcase) == expected_output

    # both will pass
    testcase1 = "Aria's phone number is +1 8888888888 must be updated."
    expected1 = "Aria's phone number is +1 XXXXXXXXXX must be updated."
    result1 = replace_pii(testcase1) == expected1

    testcase2 = "Aria's phone number is +18888888888 must be updated."
    expected2 = "Aria's phone number is +1 XXXXXXXXXX must be updated."
    result2 = replace_pii(testcase2) == expected2

    assert result1 or result2


def test_clean_text():
    foo = "?" * 110
    lines = ["Punctuate this!!!! Does it stay? Yes.",
                "This should stay. It is proper.",
                f"{foo}",
                "This should not stay It lacks punctuation",
    ]
    testcase = "\n".join(lines)
    expected_output = "\n".join([lines[0], lines[1]])

    assert clean_text(testcase) == expected_output


def test_heuristic_quality_filter():
    assert not heuristic_quality_filter("What an ass tounding discovery!"), \
        "Failed to return False for text with a bad word in it."

    assert not heuristic_quality_filter("Hello world"), \
        "Failed to return False for text with no punctuation."

    assert not heuristic_quality_filter("                    "), \
        "Failed to return False for text that is entirely whitespace."

    assert not heuristic_quality_filter("☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺ok"), \
        "Failed to return False for text that is more than 80 percent '☺'s."
    
    assert heuristic_quality_filter("Hello world! ☺"), \
        "Failed to return True for a valid input: `Hello world! ☺'"