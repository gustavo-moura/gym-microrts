from league import Match

m = Match(False, match_up=['vulcanMCTSAI', 'vulcanMCTSAI'])

## Capture video
# capture_video = True
# experiment_name = 'hello'

# if capture_video:
#     from stable_baselines3.common.vec_env import VecVideoRecorder

#     m.envs = VecVideoRecorder(
#         m.envs, f"videos/{experiment_name}", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000
#     )

##

r = m.run(1)

print(r)

import pdb; pdb.set_trace()