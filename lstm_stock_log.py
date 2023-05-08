import time
#store log info into file.

def output(log_info,level=0):
    date_file=time.strftime("%Y%m", time.localtime())
    date_full=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if level >0 :
        log_info="Error:"+log_info
    with open("./log/log"+date_file+".txt","a") as log_file:
        log_info_r=date_full+": "+log_info+"\n\r"
        log_file.write(log_info_r)