# https://github.com/ucsb-gradescope-tools/pytest_utils/blob/master/pytest_utils/decorators.py

# Decorator for setting the max score of a test
def max_score(max_score):
    def wrapper(f):
        f.max_score = max_score
        return f

    return wrapper


# Optional decorator for setting the visibility of a test
# Options: 'visible', 'hidden', 'after_due_date', 'after_published'
def visibility(visibility):
    def wrapper(f):
        f.visibility = visibility
        return f

    return wrapper


# Optional decorator for adding extra tags to a test
# Should be an array of strings
def tags(tags):
    def wrapper(f):
        f.tags = tags
        return f

    return wrapper
