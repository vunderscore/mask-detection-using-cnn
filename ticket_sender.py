import random
import time
from paho.mqtt import client as mqtt_client

broker = 'broker.emqx.io'
port = 1883
topic = 'test/topic'

client_id = 'client_1'

def connect_mqtt():
	def on_connect(client, userdata, flags, rc):
		if rc == 0:
			print('connection established')
		else:
			print('connection failed, code',rc)

	client = mqtt_client.Client(client_id)
	client.on_connect = on_connect
	client.connect(broker,port)
	return client


def publish(client):
	msg_count = 1
	while True:
		time.sleep(1)
		result = client.publish(topic,msg)
		status = result[0]
		if status == 0:
			print('message: ',msg,"sent to topic: ",topic)
		else:
			print('failed to send msg')
		msg_count += 1
		if(msg == 'exit'):
			print("connection terminated")
			break

def run():
	client = connect_mqtt()
	client.loop_start()
	publish(client)
	client.loop_stop()

run()
