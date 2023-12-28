import time
import zmq
import threading
import argparse
import redis
import asyncio
from bleak import BleakClient, BleakScanner
from datetime import datetime
import json 

# Initialize Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)
REDIS_EXPIRED = 3600
file_path = 'ambient_HOME.json'

#### Parse CLI Args ####
parser = argparse.ArgumentParser("controller")
parser.add_argument("--agent_name", help="Agent Name, eg: Spider1", type=str, default="Spider1")
args = parser.parse_args()

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

device_name = args.agent_name

pub_socket = None
sub_socket = None

def subscriber_thread():
    global sub_socket
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect("tcp://haydn.ee.ntu.edu.tw:10004")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')

def requester_thread():
    global pub_socket
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.connect("tcp://haydn.ee.ntu.edu.tw:10005")

    while True:
        # Send a message to the server and wait for a reply
        pub_socket.send_string(device_name + ": Hello")
        # message = pub_socket.recv_string()
        # print(f"Received reply from server: {message}")
        time.sleep(5)

threading.Thread(target=subscriber_thread).start()
threading.Thread(target=requester_thread).start()

service_uuid = "19B10000-E8F2-537E-4F6C-111111111111"
request_on_char_uuid = "19B10001-E8F2-537E-4F6C-111111111111"
request_off_char_uuid = "19B10001-E8F2-537E-4F6C-222222222222"

response_char_uuid = "19B10002-E8F2-537E-4F6C-111111111111"

last_decibel = ""

def notification_handler(sender, data):
    global pub_socket, device_name, last_decibel, file_path
    message = data.decode()
    message = message.replace("'", '"')

    print(f"{device_name} - Recv from Bluetooth: {message}")
    pub_socket.send_string(f"{device_name}: {message}")

    arr_msg = json.loads(message)
    print(arr_msg)
    if arr_msg['event'] == 'decibel':
        last_decibel = arr_msg['decibel']
    if arr_msg['event'] == 'prediction':
        ambient = arr_msg['ambient']
        home = arr_msg['home']
        neighbor = arr_msg['neighbor']
        # Save data to Redis with an expiration of REDIS_EXPIRED seconds (eg: 60 seconds = 1 minute)
        # Get the current time
        now = datetime.now()
        timestamp = int(now.timestamp())
        microseconds = now.microsecond

        # Combine them to form a sequence number
        sequence_number = f"{timestamp}{microseconds:06d}"

        data = {
            'decibel': last_decibel,
            'ambient': ambient,
            'home': home,
            'neighbor': neighbor
        }
        json_msg = json.dumps(data)
        print(f"Sequence Number: {sequence_number}, data: {json_msg}")

        redis_client.setex(f"bluetooth_data_{sequence_number}", REDIS_EXPIRED, json_msg)
        # Append the JSON string to the file
        # with open(file_path, 'a') as file:
        #     file.write(json_msg + '\n')

async def run():
    devices = await BleakScanner.discover()
    target_device = None
    for device in devices:
        print(device)
        if device.name == device_name:
            target_device = device
            break

    if not target_device:
        print("Device not found")
        return

    client = BleakClient(target_device.address)

    try:
        while True:
            try:
                await client.connect()
                if await client.is_connected():
                    print(f"Connected to {device_name}")

                    await client.start_notify(response_char_uuid, notification_handler)

                    # Toggle the LED state every 5 seconds
                    while await client.is_connected():
                        print("Sent: 1 ON")
                        await client.write_gatt_char(request_on_char_uuid, bytearray([0x01]))
                        await asyncio.sleep(1)

                        print("Sent: 1 OFF")
                        await client.write_gatt_char(request_off_char_uuid, bytearray([0x01]))
                        await asyncio.sleep(1)

                        print("Sent: 2 ON")
                        await client.write_gatt_char(request_on_char_uuid, bytearray([0x02]))
                        await asyncio.sleep(1)

                        print("Sent: 2 OFF")
                        await client.write_gatt_char(request_off_char_uuid, bytearray([0x02]))
                        await asyncio.sleep(1)

                        print("Sent: 3 ON")
                        await client.write_gatt_char(request_on_char_uuid, bytearray([0x03]))
                        await asyncio.sleep(1)

                        print("Sent: 3 OFF")
                        await client.write_gatt_char(request_off_char_uuid, bytearray([0x03]))
                        await asyncio.sleep(1)

                    await client.stop_notify(response_char_uuid)
                else:
                    print("Failed to connect. Retrying in 1 seconds...")
                    await asyncio.sleep(1)

            except Exception as e:
                print(f"Connection lost or failed: {e}. Reconnecting in 1 seconds...")
                await asyncio.sleep(1)

    finally:
        await client.disconnect()

asyncio.run(run())