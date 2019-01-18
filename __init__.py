# from cartpole_bullet import CartPoleBulletEnv
# from minitaur_gym_env import MinitaurBulletEnv
# from minitaur_duck_gym_env import MinitaurBulletDuckEnv
# from racecarGymEnv import RacecarGymEnv
# from racecarZEDGymEnv import RacecarZEDGymEnv
# from kukaGymEnv import KukaGymEnv
# from kukaCamGymEnv import KukaCamGymEnv

from gym.envs.registration import register
# register(
#     id="Pusher-v1",
#     entry_point="micoenv.mico_robot_env:MicoEnv",
#     kwargs={
#         "randomize_arm": True,
#         "randomize_camera": True,
#         "randomize_textures": True,
#         "randomize_objects": True,
#         "normal_textures": True,
#         "done_after": 300,
#         'target_in_the_air': False,
#         "has_object": True,
#         "reward_type": "positive",
#         "observation_type": "pixels",
#
#     }
# )

register(
    id="Grasper-v1",
    entry_point="pybullet_envs.bullet.kuka_diverse_object_gym_env:KukaDiverseObjectEnv"
)