import envpool
import numpy

import cv2

NUM_ENVS = 4
STEPS = 1000

# 1. Initialize environment with rgb_array render mode
env = envpool.make(
    "DogRun-v1",
    env_type="gymnasium",
    num_envs=NUM_ENVS,
    render_mode="rgb_array",
    render_width=480,
    render_height=480,
    seed=42,
)

obs, info = env.reset()

print(obs)

while True:
    # Sample batched random actions
    actions = numpy.random.uniform(
        low=env.action_space.low,
        high=env.action_space.high,
        size=(NUM_ENVS, *env.action_space.shape),
    ).astype(numpy.float32)

    next_obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated[0]:
        env[0].reset()

    # env.render() returns a batched array: shape (N, H, W, 3)
    # env_ids=[0] extracts frame for the first environment in the pool
    frame = env.render(env_ids=[0])  # Shape: (1, 480, 480, 3) 

    frame = frame[0]

    #frame = numpy.swapaxes(frame, 0, 2)
    frame = numpy.array(frame/255.0, dtype=numpy.float32)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)



    cv2.imshow("image", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

env.close()
