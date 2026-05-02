import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def send_email(subject: str, message: str):
    """
    Sends an email notification about the pipeline status.
    Requires environment variables: SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL
    """
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port = os.environ.get("SMTP_PORT", "587")
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    recipient_email = os.environ.get("RECIPIENT_EMAIL")

    if not all([smtp_server, sender_email, sender_password, recipient_email]):
        print("Email configuration incomplete (SMTP_SERVER, SENDER_EMAIL, etc.), skipping email notification.")
        # Print the message to console so it's not lost
        print(f"--- EMAIL NOTIFICATION (SKIPPED) ---\nSubject: {subject}\nMessage: {message}\n-----------------------------------")
        return

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    body = f"{message}\n\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Email notification sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email notification: {e}")
