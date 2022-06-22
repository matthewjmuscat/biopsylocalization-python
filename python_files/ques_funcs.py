# Stores question functions

def ask_ok(prompt, retries=4, reminder='Please try again!'):
    #This function defines a generic yes or no question
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