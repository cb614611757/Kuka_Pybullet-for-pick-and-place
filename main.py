import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
import gym
import numpy as np
import pdb
import cv2
from kuka_diverse_object_gym_env import KukaDiverseObjectEnv

import demo_policies as demo

save_path = './data/train_image/'
def main():
    # environment = KukaCamGymEnv(renders=True, isDiscrete=False)
    environment = KukaDiverseObjectEnv(renders=True, isDiscrete=False)
    demo_policy_object = demo.policies['grasper']()
    for i in range(1000):
        done = False
        index = 0
        environment._reset()
        # print(action)
        # pdb.set_trace()
        while (not done):
            action = []
            low_state = environment.state_vector()
            action = demo_policy_object._choose_action(low_state)
            print('action:',action)
            state, reward, done, info = environment.next_step(action)
            index += 1
            print('step: ', index)
            print('reward: ', reward)

            image = environment._get_observation()

            image_name = os.path.join(os.path.abspath(save_path), str(i) + str(index) + '.jpg')

            cv2.imwrite(image_name, image)


if __name__=="__main__":
    main()
