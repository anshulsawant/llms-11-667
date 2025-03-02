import math
from calculator.utils import can_use_calculator, use_calculator, extract_label

from pytest_utils.decorators import max_score


@max_score(5)
def test_can_use_calculator():
    assert not can_use_calculator("")
    assert not can_use_calculator("hello world")
    assert not can_use_calculator("my mom is a math teacher")
    assert not can_use_calculator("Q: 123456")
    assert not can_use_calculator("<< 1+2 >> ")
    assert can_use_calculator("<<1+2>>")
    assert can_use_calculator("<< 1+2 >>")
    assert can_use_calculator("</1231??>>")


@max_score(5)
def test_use_calculator():
    assert use_calculator("") == ""
    assert (
        use_calculator("Q: How many mugs do I have? A: <<3+2+1>>")
        == "Q: How many mugs do I have? A: <<3+2+1>>6"
    )
    assert (
        use_calculator("Q: How many dogs do I have? A: <<3*2*1>>")
        == "Q: How many dogs do I have? A: <<3*2*1>>6"
    )
    assert use_calculator("A: <<>>") == "A: <<>>"
    assert use_calculator("A: <<z9w0e>>") == "A: <<z9w0e>>"
    div_ans = use_calculator("very difficult question: <<1/2+1>>")
    assert math.isclose(extract_label(div_ans), 1.5)
