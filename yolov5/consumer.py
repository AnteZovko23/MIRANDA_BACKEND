import pika
import sys
import os
import base64
import io
def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='commands')

    def callback(ch, method, properties, body):
        print(" [x] Received %r" % body)
        
        try:
            print(body)
            byte_array = io.BytesIO(body)
            with open('output.3gp', 'wb') as f:
                print(byte_array.getbuffer())
                f.write(byte_array.getbuffer())
        except Exception as e:
            print("Failed to write to file")
            print(e)

    channel.basic_consume(queue='commands', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)