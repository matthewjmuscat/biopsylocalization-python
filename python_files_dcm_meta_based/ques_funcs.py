# Stores question functions

def ask_ok(prompt, retries=4, reminder='Please try again!'):
    """
    This function defines a generic yes or no question.
    """
    while True:
        ok = input(prompt + ' [yes/no]\n')
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)


def ask_to_continue(prompt, retries=4, reminder='Please try again!'):
    """
    This function defines a generic continue confirmation.
    """
    while True:
        ok = input(prompt + ' [enter]\n')
        if ok in ('\r'):
            return True
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)

def multi_choice_question(prompt, retries=4, reminder='Please try again!'):
    """
    This function defines a generic yes or no question.
    """
    while True:
        ans = input(prompt + ' [1/0]\n')
        if ans in ('1', 'one'):
            return True
        if ans in ('0', 'zero'):
            return False
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)

def ask_for_float_question(prompt, retries=4, reminder='Please try again!'):
    """
    This function defines a generic yes or no question.
    """
    while True:
        ans = input(prompt + '\n')
        if isfloat(ans):
            return float(ans)
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False