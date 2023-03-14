import pika

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='phone_wrist_info')

# Listen for messages on the queue
def callback(ch, method, properties, body):
    print(" [x] Received %r" % body)
    
