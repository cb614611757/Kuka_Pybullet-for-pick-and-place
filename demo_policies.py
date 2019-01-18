import numpy as np
import pdb


class DemoPolicy(object):
    def choose_action(self, state):
        return np.clip(self._choose_action(state), -0.5, 0.5)

    def reset(self):
        raise Exception("Not implemented")


class Waypoints_grasper(DemoPolicy):
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.currentWaypoint = 0
        self.last_angle = 1

    @staticmethod
    def go_to_waypoint(grip_pos, waypoint):
        print('waypoint:',waypoint)
        print('grip_pos:',grip_pos)
        print('div', np.linalg.norm(grip_pos - waypoint))
        return (
            np.concatenate((waypoint - grip_pos, [0.0])),
            np.linalg.norm(grip_pos - waypoint) < 0.1,
        )

    def _choose_action(self, state):

        grip_pos = state[0:3]
        print('currentWaypoint:', self.currentWaypoint)
        action, done = self.go_to_waypoint(
            grip_pos, self.waypoints[min(self.currentWaypoint,
                                         len(self.waypoints) - 1)])
        # print(action)

        # print(action)
        if done:
            self.currentWaypoint += 1
            if self.currentWaypoint == 2:
                self.last_angle = 0.0
            elif self.currentWaypoint == 5:
                self.last_angle = 1

        action = np.concatenate((action, [self.last_angle]))

        return action

    def done(self):
        return self.currentWaypoint >= len(self.waypoints)


class Grasper(DemoPolicy):
    def __init__(self):
        self.policy = None

    def _choose_action(self, state):
        if not self.policy:
            print(state)
            goal_pos = state[3:6]
            object_pos = state[6:9]
            object_pos[1] -= 0
            object_pos[2] -= 0.14
            print('goal:', goal_pos)
            print('object:', object_pos)
            # object_rel = object_pos - goal_pos
            # behind_obj = object_pos + object_rel / \
            #     np.linalg.norm(object_rel) * 0.06
            # behind_obj[2] = 0.03
            waypoints = [np.concatenate([object_pos[:2], [0.3]]), object_pos, np.concatenate([object_pos[:2], [0.3]]),
                         np.concatenate([goal_pos[:2], [0.4]]), np.concatenate([goal_pos[:2], [0.1]])]

            self.policy = Waypoints_grasper(waypoints)
        action = self.policy._choose_action(state)

        if self.policy.done():
            self.policy = None
        # action += np.random.normal([0,0,0,0], 0.15)
        return action/3

    def reset(self):
        self.policy = None



class ArmData(object):
    def __init__(self, data):
        assert data.shape == (13, )
        self.grip_pos, self.grip_velp, self.gripper_state, self.isGrasping, self.goalPosition, self.goalGripper = (
            data[0:3],
            data[3:6],
            data[6:8],
            data[8],
            data[9:12],
            data[12],
        )


policies = {
    # "pusher": Pusher,
    "None": None,
    "grasper": Grasper,
}
