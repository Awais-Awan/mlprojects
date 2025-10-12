import sys


def error_message_details(error, error_details:sys): #custom excetion to call whenever an error occur
   _,_,exc_tb = error_details.exc_info() #this function return all the information for an error
   file_name = exc_tb.tb_frame.f_code.co_filename
   error_message = "Error occured in python scripts name [{0}]\nLine number [{1}]\nError message [{2}]".format(file_name,exc_tb.tb_lineno,error)
   
   return error_message


class CustomException(Exception):
   def __init__(self,erorr_message,error_detail:sys):
      super().__init__(erorr_message)
      self.error_messsage = error_message_details(erorr_message,error_details=error_detail)
    
   def __str__(self):
       return self.error_messsage
   


