import pprint
import smtplib
from email.mime.text import MIMEText



def send_email_notification(subject, body, dest_address='riccardo.delchiaro@gmail.com', verbose=True):
    usr="rdc.micc@gmail.com"
    pwd="M1cc!pwd!"
    server = 'smtp.gmail.com'
    port = 465
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = usr
    msg['To'] = dest_address

    if verbose:
        print("Connecting to server: {} (port: {})".format(server, port))
    s = smtplib.SMTP_SSL(server, port)
    s.ehlo()
    s.login(usr,pwd)

    s.sendmail(usr, [dest_address], msg.as_string())
    s.close()
    if verbose:
        print("")
        print("-----------------------------")
        print("Email to:   " + dest_address)
        print("Email from: " + usr)
        print("Email subject: " + subject)
        print("-----------------------------")

        print("Email body: ")
        print("-----------------------------")
        pprint.pprint(body)

