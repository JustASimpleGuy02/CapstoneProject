from PyQt5.QtCore import QRunnable, pyqtSlot
import traceback
import logging

class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''
    
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.setAutoDelete(True)
        
        
    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            logging.debug(f"Worker -- run: {e}")
            traceback.print_exc()