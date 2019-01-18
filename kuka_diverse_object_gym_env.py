from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import random
import os
from gym import spaces
import time
import pybullet as p
import kuka
import numpy as np
import pybullet_data
import pdb
import distutils.dir_util
import glob
import gym
import perlin_noise as noise
import uuid
import sys

tmp_dir = os.path.dirname(sys.modules['__main__'].__file__) + "/tmp"
class KukaDiverseObjectEnv(KukaGymEnv):
  """Class for Kuka environment with diverse objects.

  In each episode some objects are chosen from a set of 1000 diverse objects.
  These 1000 objects are split 90/10 into a train and test set.
  """

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=80,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps=30,
               dv=0.06,
               removeHeightHack=False,
               blockRandom=0.15,
               cameraRandom=0,
               width=280,
               height=280,
               numObjects=5,
               isTest=False):
    """Initializes the KukaDiverseObjectEnv. 

    Args:
      urdfRoot: The diretory from which to load environment URDF's.
      actionRepeat: The number of simulation steps to apply for each action.
      isEnableSelfCollision: If true, enable self-collision.
      renders: If true, render the bullet GUI.
      isDiscrete: If true, the action space is discrete. If False, the
        action space is continuous.
      maxSteps: The maximum number of actions per episode.
      dv: The velocity along each dimension for each action.
      removeHeightHack: If false, there is a "height hack" where the gripper
        automatically moves down for each action. If true, the environment is
        harder and the policy chooses the height displacement.
      blockRandom: A float between 0 and 1 indicated block randomness. 0 is
        deterministic.
      cameraRandom: A float between 0 and 1 indicating camera placement
        randomness. 0 is deterministic.
      width: The image width.
      height: The observation image height.
      numObjects: The number of objects in the bin.
      isTest: If true, use the test set of objects. If false, use the train
        set of objects.
    """

    self._isDiscrete = isDiscrete
    self._timeStep = 1./240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40
    self._dv = dv
    self._p = p
    self._removeHeightHack = removeHeightHack
    self._blockRandom = blockRandom
    self._cameraRandom = cameraRandom
    self._width = width
    self._height = height
    self._numObjects = numObjects
    self._isTest = isTest
    self.envId = uuid.uuid4()

    if self._renders:
      self.cid = p.connect(p.SHARED_MEMORY)
      if (self.cid<0):
        self.cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.5,200,-40,[0.52,-0.2,-0.33])
    else:
      self.cid = p.connect(p.DIRECT)
    self._seed()

    if (self._isDiscrete):
      if self._removeHeightHack:
        self.action_space = spaces.Discrete(9)
      else:
        self.action_space = spaces.Discrete(7)
    else:
      self.action_space = spaces.Box(low=-1, high=1, shape=(3,))  # dx, dy, da
      if self._removeHeightHack:
        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(4,))  # dx, dy, dz, da
    self.viewer = None

  def _reset(self):
    """Environment reset called at the beginning of an episode.
    """
    # Set the camera settings.

    look = [0.23, 0.2, 0.54]
    distance = 2.
    pitch = -36 + self._cameraRandom*np.random.uniform(-3, 3)
    yaw = 245 + self._cameraRandom*np.random.uniform(-3, 3)
    roll = 0
    self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
      look, distance, yaw, pitch, roll, 2)
    fov = 20. + self._cameraRandom*np.random.uniform(-2, 2)
    aspect = self._width / self._height
    near = 0.01
    far = 20
    self._proj_matrix = p.computeProjectionMatrixFOV(
      fov, aspect, near, far)

    # x = np.random.normal(-1.05, 0.04, 1)
    # z = np.random.normal(0.68, 0.04, 1)
    # lookat_x = np.random.normal(0.1, 0.02, 1)
    # pos = [x, 0, z]
    # lookat = [lookat_x, 0, 0]
    # print(pos)
    # vec = [-0.5, 0, 1]
    # self._view_matrix = p.computeViewMatrix(pos, lookat, vec)
    # fov = np.random.normal(45, 2, 1)
    # self._proj_matrix = p.computeProjectionMatrixFOV(
    #   fov=fov, aspect=4. / 3., nearVal=0.01, farVal=2.5)


    self._attempted_grasp = False
    self._env_step = 0
    self.terminated = 0

    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    plane = p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])

    table = p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)
    # self.planeId = plane[0]
    # self.tableId = table[0]
    p.setGravity(0,0,-10)

    self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)


    self.goal = [0.6,0.4,-0.19]
    self._action_angle=0

    direction = np.array([
      np.random.choice([
        np.random.random_integers(-20, -5),
        np.random.random_integers(5, 20),
      ]),
      np.random.choice([
        np.random.random_integers(-20, -5),
        np.random.random_integers(5, 20),
      ]),
      np.random.random_integers(70, 100),
    ])

    self.light = {
      "diffuse": np.random.uniform(0.4, 0.6),
      "ambient": np.random.uniform(0.4, 0.6),
      "spec": np.random.uniform(0.4, 0.7),
      "dir": direction,
      "col": np.random.uniform([0.9, 0.9, 0.9], [1, 1, 1]),
    }


    wood_color = np.random.normal([170, 150, 140], 8)
    wall_color = np.random.normal([230, 240, 250], 8)

    tex1 = p.loadTexture(
      noise.createAndSave(
        tmp_dir + "/wall-{}.png".format(self.envId),
        "cloud",
        wall_color,
        ))
    tex2 = p.loadTexture(
      noise.createAndSave(
        tmp_dir + "/table-{}.png".format(self.envId),
        "cloud",
        wood_color,
        ))
    p.changeVisualShape(plane, -1, textureUniqueId=tex2)
    p.changeVisualShape(table, -1, textureUniqueId=tex1)

    self._envStepCounter = 0

    p.stepSimulation()

    # Choose the objects in the bin.
    urdfList = self._get_random_object(
      self._numObjects, self._isTest)
    self._objectUids,self.obj_pos = self._randomly_place_objects(urdfList)
    print('_objectUids:',self._objectUids)
    self._observation = self._get_observation()
    return np.array(self._observation)

  def state_vector(self):
    grip_state = p.getLinkState(
      self._kuka.kukaUid, 6, computeLinkVelocity=1)
    self.grip_pos = np.array(grip_state[0])

    gripper_state = [
      p.getJointState(self._kuka.kukaUid, 8)[0],
      p.getJointState(self._kuka.kukaUid, 11)[0],
    ]
    low_dim=np.concatenate([
      self.grip_pos.copy(),
      self.goal.copy(),
      self.obj_pos.copy(),
      gripper_state.copy(),
    ])
    return low_dim.astype(np.float32)

  def _randomly_place_objects(self, urdfList):
    """Randomly places the objects in the bin.

    Args:
      urdfList: The list of urdf files to place in the bin.

    Returns:
      The list of object unique ID's.
    """


    # Randomize positions of each object urdf.
    objectUids = []
    for urdf_name in urdfList:
      xpos = 0.6 +self._blockRandom*random.random()
      ypos = -0.15+self._blockRandom*(random.random())
      angle = np.pi/2 + self._blockRandom * np.pi * random.random()
      orn = p.getQuaternionFromEuler([0, 0, angle])
      urdf_path = os.path.join(self._urdfRoot, urdf_name)
      basepose=[0.6, -0.1, 0.15]
      uid = p.loadURDF(urdf_path,  [xpos,ypos,0.15],
                       [orn[0], orn[1], orn[2], orn[3]])
      # time.sleep(self._timeStep*20)
      for _ in range(500):
        p.stepSimulation()
      objectUids.append(uid)
    return objectUids, basepose

  def _get_observation(self):
    """Return the observation as an image.
    """
    img_arr = p.getCameraImage(width=self._width,
                               height=self._height,
                               viewMatrix=self._view_matrix,
                               projectionMatrix=self._proj_matrix,
                               shadow=1,
                               lightAmbientCoeff=self.light["ambient"],
                               lightDiffuseCoeff=self.light["diffuse"],
                               lightSpecularCoeff=self.light["spec"],
                               lightDirection=self.light["dir"],
                               lightColor=self.light["col"])
    rgb = img_arr[2]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
    return np_img_arr[:, :, :3]

  def next_step(self, action):
    if(action[4] > 0.1) and (self._action_angle != action[4]):
      self._action_angle = action[4]
      for i in range(100):
        action_angle = [0, 0, 0, 0, action[4]/100*i]
        # time.sleep(self._timeStep)
        self._kuka.applyAction(action_angle)
        for _ in range(2):
          time.sleep(self._timeStep)
          p.stepSimulation()
    elif (action[4] < 0.1) and (self._action_angle != action[4]):
      self._action_angle = action[4]
      for i in range(50):
        action_angle = [0, 0, 0, 0, action[4]]
        # time.sleep(self._timeStep)
        self._kuka.applyAction(action_angle)
        for _ in range(10):
          time.sleep(self._timeStep*2)
          p.stepSimulation()

    time.sleep(self._timeStep * 240)


    for i in range(1):
      action_x = [action[0],0, 0, 0, action[4]]
      # time.sleep(self._timeStep)
      self._kuka.applyAction(action_x)
      for _ in range(30):
        time.sleep(self._timeStep)
        p.stepSimulation()

    for i in range(1):
      action_y = [0, action[1],0,  0, action[4]]
      # time.sleep(self._timeStep)
      self._kuka.applyAction(action_y)
      for _ in range(30):
        time.sleep(self._timeStep)
        p.stepSimulation()

    for i in range(1):
      action_z = [0, 0, action[2], 0, action[4]]
      # time.sleep(self._timeStep)
      self._kuka.applyAction(action_z)
      for _ in range(30):
        time.sleep(self._timeStep)
        p.stepSimulation()

    observation = self._get_observation()
    near_goal=self.goal
    near_goal[2] = 0.1
    done = (np.linalg.norm(self.grip_pos - near_goal) < 0.1)

    reward = self._reward()
    debug = {
      'grasp_success': self._graspSuccess
    }
    return observation, reward, done, debug

  def _reward(self):
    """Calculates the reward for the episode.

    The reward is 1 if one of the objects is above height .2 at the end of the
    episode.
    """
    reward = 0
    self._graspSuccess = 0
    for uid in self._objectUids:
      pos, _ = p.getBasePositionAndOrientation(
        uid)
      # If any block is above height, provide reward.
      if pos[1] > 0.1:
        self._graspSuccess += 1
        reward = 1
        break
    return reward

  def _termination(self):
    """Terminates the episode if we have tried to grasp or if we are above
    maxSteps steps.
    """
    return self._attempted_grasp or self._env_step >= self._maxSteps

  def _get_random_object(self, num_objects, test):
    """Randomly choose an object urdf from the random_urdfs directory.

    Args:
      num_objects:
        Number of graspable objects.

    Returns:
      A list of urdf filenames.
    """
    if test:
      urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*0/*.urdf')
    else:
      urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*[^0]/*.urdf')
    found_object_directories = glob.glob(urdf_pattern)
    total_num_objects = len(found_object_directories)
    selected_objects = np.random.choice(np.arange(total_num_objects),
                                        num_objects)
    selected_objects_filenames = []
    for object_index in selected_objects:
      selected_objects_filenames += [found_object_directories[object_index]]
    return selected_objects_filenames

