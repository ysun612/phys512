class Log:
    """
    A class that implements basic logging
    """
    def __init__(self):
        """
        Initialization function
        """
        self.log = ''

    def append(self, *args, do_print=True):
        """
        Print and add log to the object
        :param args: things that want to save
        :param do_print: print
        :return: None
        """
        if do_print:
            # First we print all the arguments
            print(*args)
        # For each arg, we append it to self.log
        for a in args:
            self.log += str(a)
            # Add a space to the end, similar to print
            self.log += ' '
        # Add a new line at the end, also similar to print
        self.log += '\n'

    def save(self, file):
        """
        Save all the log to file
        :param file: file name
        :return: None
        """
        with open(file, 'w') as f:
            f.write(self.log)
