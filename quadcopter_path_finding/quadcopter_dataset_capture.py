from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import cv2
import time
import os

def count_file(path):
    files = os.listdir(path)
    num_files = sum(1 for f in files if os.path.isfile(os.path.join(path, f)))
    return num_files

client = RemoteAPIClient()
sim = client.require('sim')

print('Simulation started')
sim.startSimulation()

quadcopter_target = sim.getObject('/target')
camera = sim.getObject('/Quadcopter/visionSensor')

position = sim.getObjectPosition(quadcopter_target, -1)
x, y, z = position

delta = 0.1 

forward_dir = "C:/Users/ASUS/Documents/Code/CoppeliaSims_py/quadcopter_path_finding/train_data/forward"
os.makedirs(forward_dir, exist_ok=True)
up_dir = "C:/Users/ASUS/Documents/Code/CoppeliaSims_py/quadcopter_path_finding/train_data/up"
os.makedirs(up_dir, exist_ok=True)
down_dir = "C:/Users/ASUS/Documents/Code/CoppeliaSims_py/quadcopter_path_finding/train_data/down"
os.makedirs(down_dir, exist_ok=True)
left_dir = "C:/Users/ASUS/Documents/Code/CoppeliaSims_py/quadcopter_path_finding/train_data/left"
os.makedirs(left_dir, exist_ok=True)
right_dir = "C:/Users/ASUS/Documents/Code/CoppeliaSims_py/quadcopter_path_finding/train_data/right"
os.makedirs(right_dir, exist_ok=True)

forward_ct = count_file(forward_dir)
up_ct = count_file(up_dir)
down_ct = count_file(down_dir)
left_ct = count_file(left_dir)
right_ct = count_file(right_dir)

try:
    while True:
        img, [resX, resY] = sim.getVisionSensorImg(camera)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0)
        img = cv2.resize(img, (512,512))
        cv2.imshow('Quadcopter View', img)

        com = cv2.waitKey(1)

        if com == ord('q'):
            print('Program stopped')
            break
        elif com == ord('w'):
            x += delta
            filename = os.path.join(forward_dir, f"forward_{forward_ct}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            forward_ct += 1
        elif com == ord('s'):
            x -= delta
        elif com == ord('a'):
            y += delta
            filename = os.path.join(left_dir, f"left_{left_ct}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            left_ct += 1
        elif com == ord('d'):
            y -= delta
            filename = os.path.join(right_dir, f"right_{right_ct}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            right_ct += 1
        elif com == ord(' '):
            z += delta
            filename = os.path.join(up_dir, f"up_{up_ct}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            up_ct += 1
        elif com == ord('z'):
            z -= delta
            filename = os.path.join(down_dir, f"down_{down_ct}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            down_ct += 1
                    
        sim.setObjectPosition(quadcopter_target, -1, [x, y, z])

        time.sleep(0.01)

except Exception as e:
    print(f"❌ Error: {e}")

cv2.destroyAllWindows()
sim.stopSimulation()
print("Simulation stopped")
