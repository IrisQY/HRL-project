import numpy as np
import math
import gym
from gym import spaces
from gym.utils import seeding

class Environment(object):
    """
    generic class for environments
    """
    def reset(self):
        """
        returns initial observation
        """
        pass

    def step(self, action):
        """
        returns (observation, termination signal)
        """
        pass


class MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5

        self.force=0.001
        self.gravity=0.0025

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action-1)*self.force + math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position)
        reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def get_keys_to_action(self):
        return {():1,(276,):0,(275,):2,(275,276):1} #control with left and right arrow keys

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

'''
class CartpoleEnv(Environment):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. 
        The pole either starts in an upright position or a downward position. 
        The goal is to keep the pole upright within some thresholds by applying force to the cart. 
        
        Modified from the cartpole environment on OpenAI gym. 

    Observation: 
        0   Cart Position
        1   Cart Velocity
        2   Pole Angle
        3   Pole Velocity At Tip
        
    Actions:
        0   Push cart to the left
        1   No action
        2   Push cart to the right

    Starting State:
        Cart position, velocity, and angular velocity are drawn uniformly from [-0.05, 0.05]. 
        Pole angle is drawn uniformly from [-0.05, 0.05] if starting upright, and from 
        [pi-0.05, pi+0.05] if starting downwards. 

    Episode Termination:
        Cart Position is more than some threshold away from 0. 
    """

    def __init__(self, swing_up=False, timescale=0.02):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = timescale  # seconds between state updates
        self.state = None #(x, x_dot, theta, theta_dot)
        self.x_threshold = 5
        self.swing_up = swing_up # determines pole's initial position

    def step(self, action):
        assert action in [0, 1, 2], "invalid action"
        x, x_dot, theta, theta_dot = self.state
        force = (action - 1) * self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta = np.remainder(theta, 2*math.pi)
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)
        done = bool((x<-self.x_threshold) or (x>self.x_threshold))
        
        return np.array(self.state), done

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        if self.swing_up:
            self.state[2] += math.pi

        return np.array(self.state)
'''

### test
'''
if __name__=='__main__':
    swing_up = False
    nsteps = 50
    print("swing up is %r, number of steps is %d" % (swing_up, nsteps))

    np.random.seed(0)
    env = CartpoleEnv(swing_up=swing_up)
    obs = env.reset()
    t = 0
    print("t=%d, x %.2f, theta %.2f, theta_dot %.2f"
        %(t, obs[0], obs[2], obs[3]))

    done = False
    while not done:
        action = np.random.randint(3)
        obs, done = env.step(action)
        t += 1
        print("t=%d, action %d, x %.2f, x_dot %.2f, theta %.2f, theta_dot %.2f, done %r" 
            %(t, action, obs[0], obs[1], obs[2], obs[3], done))
        done = done or t==nsteps
'''

